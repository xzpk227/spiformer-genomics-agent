import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as efs from "aws-cdk-lib/aws-efs";
import * as servicediscovery from "aws-cdk-lib/aws-servicediscovery";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as logs from "aws-cdk-lib/aws-logs";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

interface EcsStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  backendRepo: ecr.Repository;
  frontendRepo: ecr.Repository;
  spliformerRepo: ecr.Repository;
  reportsBucket: s3.Bucket;
}

export class EcsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: EcsStackProps) {
    super(scope, id, props);

    const { vpc, backendRepo, frontendRepo, spliformerRepo, reportsBucket } = props;

    // ── Secrets ────────────────────────────────────────────────────────────
    const openaiSecret = secretsmanager.Secret.fromSecretNameV2(
      this, "OpenAiSecret", "genomics/openai-api-key"
    );
    const pineconeSecret = secretsmanager.Secret.fromSecretNameV2(
      this, "PineconeSecret", "genomics/pinecone-api-key"
    );
    const ncbiSecret = secretsmanager.Secret.fromSecretNameV2(
      this, "NcbiSecret", "genomics/ncbi-api-key"
    );

    // ── Cluster ────────────────────────────────────────────────────────────
    const cluster = new ecs.Cluster(this, "Cluster", {
      vpc,
      clusterName: "genomics-cluster",
      containerInsights: true,
    });

    // ── Cloud Map namespace for internal service discovery ─────────────────
    const namespace = new servicediscovery.PrivateDnsNamespace(this, "Namespace", {
      name: "genomics.local",
      vpc,
    });

    // ── ALB ────────────────────────────────────────────────────────────────
    const alb = new elbv2.ApplicationLoadBalancer(this, "Alb", {
      vpc,
      internetFacing: true,
      loadBalancerName: "genomics-alb",
    });

    const listener = alb.addListener("HttpListener", {
      port: 80,
      defaultAction: elbv2.ListenerAction.fixedResponse(404, {
        contentType: "text/plain",
        messageBody: "Not found",
      }),
    });

    // ── EFS for reference genomes ──────────────────────────────────────────
    const genomeFsSg = new ec2.SecurityGroup(this, "GenomeFsSg", { vpc });

    const genomeFs = new efs.FileSystem(this, "GenomeFs", {
      vpc,
      securityGroup: genomeFsSg,
      // Explicitly place mount targets in private subnets so Fargate tasks can reach them
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecyclePolicy: efs.LifecyclePolicy.AFTER_90_DAYS,
      performanceMode: efs.PerformanceMode.GENERAL_PURPOSE,
      encrypted: true,
    });

    const genomeAccessPoint = genomeFs.addAccessPoint("GenomeAccessPoint", {
      path: "/ref",
      createAcl: { ownerGid: "0", ownerUid: "0", permissions: "755" },
      posixUser: { gid: "0", uid: "0" },
    });

    // ── Spliformer ─────────────────────────────────────────────────────────
    const spliformerTaskRole = new iam.Role(this, "SpliformerTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    reportsBucket.grantReadWrite(spliformerTaskRole);
    spliformerTaskRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite", "elasticfilesystem:ClientRootAccess"],
      resources: [genomeFs.fileSystemArn],
    }));

    const spliformerTaskDef = new ecs.FargateTaskDefinition(this, "SpliformerTaskDef", {
      memoryLimitMiB: 16384,
      cpu: 4096,
      taskRole: spliformerTaskRole,
      volumes: [{
        name: "genome-ref",
        efsVolumeConfiguration: {
          fileSystemId: genomeFs.fileSystemId,
          transitEncryption: "ENABLED",
          authorizationConfig: {
            accessPointId: genomeAccessPoint.accessPointId,
            iam: "ENABLED",
          },
        },
      }],
    });

    const spliformerContainer = spliformerTaskDef.addContainer("spliformer", {
      image: ecs.ContainerImage.fromEcrRepository(spliformerRepo, "latest"),
      portMappings: [{ containerPort: 5001 }],
      environment: {
        REPORTS_BUCKET: reportsBucket.bucketName,
        GENOME_DIR: "/ref",
      },
      logging: ecs.LogDrivers.awsLogs({
        logGroup: new logs.LogGroup(this, "SpliformerLogs", {
          logGroupName: "/ecs/genomics-spliformer",
          retention: logs.RetentionDays.ONE_WEEK,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        streamPrefix: "spliformer",
      }),
    });

    spliformerContainer.addMountPoints({
      containerPath: "/ref",
      sourceVolume: "genome-ref",
      readOnly: false,
    });

    const spliformerSg = new ec2.SecurityGroup(this, "SpliformerSg", { vpc });
    // Allow Spliformer tasks to reach EFS mount targets on NFS port
    genomeFsSg.connections.allowFrom(spliformerSg, ec2.Port.tcp(2049));

    const spliformerService = new ecs.FargateService(this, "SpliformerService", {
      cluster,
      taskDefinition: spliformerTaskDef,
      desiredCount: 1,
      securityGroups: [spliformerSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      serviceName: "genomics-spliformer",
      cloudMapOptions: {
        name: "spliformer",
        cloudMapNamespace: namespace,
        dnsRecordType: servicediscovery.DnsRecordType.A,
      },
    });

    // ── Backend ────────────────────────────────────────────────────────────
    const backendTaskRole = new iam.Role(this, "BackendTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    reportsBucket.grantReadWrite(backendTaskRole);

    const backendTaskDef = new ecs.FargateTaskDefinition(this, "BackendTaskDef", {
      memoryLimitMiB: 2048,
      cpu: 1024,
      taskRole: backendTaskRole,
    });

    backendTaskDef.addContainer("backend", {
      image: ecs.ContainerImage.fromEcrRepository(backendRepo, "latest"),
      portMappings: [{ containerPort: 8000 }],
      environment: {
        PINECONE_INDEX_NAME: "genomics-literature",
        REPORTS_BUCKET: reportsBucket.bucketName,
        // Spliformer reachable via Cloud Map DNS
        SPLIFORMER_URL: "http://spliformer.genomics.local:5001",
      },
      secrets: {
        OPENAI_API_KEY: ecs.Secret.fromSecretsManager(openaiSecret),
        PINECONE_API_KEY: ecs.Secret.fromSecretsManager(pineconeSecret),
        NCBI_API_KEY: ecs.Secret.fromSecretsManager(ncbiSecret),
      },
      logging: ecs.LogDrivers.awsLogs({
        logGroup: new logs.LogGroup(this, "BackendLogs", {
          logGroupName: "/ecs/genomics-backend",
          retention: logs.RetentionDays.ONE_WEEK,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        streamPrefix: "backend",
      }),
    });

    const backendSg = new ec2.SecurityGroup(this, "BackendSg", { vpc });
    // Allow backend → spliformer
    spliformerSg.connections.allowFrom(backendSg, ec2.Port.tcp(5001));

    const backendService = new ecs.FargateService(this, "BackendService", {
      cluster,
      taskDefinition: backendTaskDef,
      desiredCount: 1,
      securityGroups: [backendSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      serviceName: "genomics-backend",
    });

    const backendScaling = backendService.autoScaleTaskCount({ minCapacity: 1, maxCapacity: 4 });
    backendScaling.scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: 70,
      scaleInCooldown: cdk.Duration.seconds(60),
      scaleOutCooldown: cdk.Duration.seconds(60),
    });

    listener.addAction("BackendRoutes", {
      priority: 10,
      conditions: [
        elbv2.ListenerCondition.pathPatterns(["/chat", "/health", "/reports/*", "/visualize"]),
      ],
      action: elbv2.ListenerAction.forward([
        new elbv2.ApplicationTargetGroup(this, "BackendTg", {
          vpc,
          port: 8000,
          protocol: elbv2.ApplicationProtocol.HTTP,
          targets: [backendService],
          healthCheck: { path: "/health", interval: cdk.Duration.seconds(30) },
        }),
      ]),
    });

    // ── Frontend ───────────────────────────────────────────────────────────
    const frontendTaskDef = new ecs.FargateTaskDefinition(this, "FrontendTaskDef", {
      memoryLimitMiB: 512,
      cpu: 256,
    });

    frontendTaskDef.addContainer("frontend", {
      image: ecs.ContainerImage.fromEcrRepository(frontendRepo, "latest"),
      portMappings: [{ containerPort: 80 }],
      logging: ecs.LogDrivers.awsLogs({
        logGroup: new logs.LogGroup(this, "FrontendLogs", {
          logGroupName: "/ecs/genomics-frontend",
          retention: logs.RetentionDays.ONE_WEEK,
          removalPolicy: cdk.RemovalPolicy.DESTROY,
        }),
        streamPrefix: "frontend",
      }),
    });

    const frontendSg = new ec2.SecurityGroup(this, "FrontendSg", { vpc });
    const frontendService = new ecs.FargateService(this, "FrontendService", {
      cluster,
      taskDefinition: frontendTaskDef,
      desiredCount: 1,
      securityGroups: [frontendSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      serviceName: "genomics-frontend",
    });

    listener.addAction("FrontendRoute", {
      priority: 100,
      conditions: [elbv2.ListenerCondition.pathPatterns(["/*"])],
      action: elbv2.ListenerAction.forward([
        new elbv2.ApplicationTargetGroup(this, "FrontendTg", {
          vpc,
          port: 80,
          protocol: elbv2.ApplicationProtocol.HTTP,
          targets: [frontendService],
          healthCheck: { path: "/", interval: cdk.Duration.seconds(30) },
        }),
      ]),
    });

    // Allow ALB → services
    backendSg.connections.allowFrom(alb, ec2.Port.tcp(8000));
    frontendSg.connections.allowFrom(alb, ec2.Port.tcp(80));

    // ── Outputs ────────────────────────────────────────────────────────────
    new cdk.CfnOutput(this, "AppUrl", { value: `http://${alb.loadBalancerDnsName}` });
    new cdk.CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new cdk.CfnOutput(this, "GenomeFileSystemId", {
      value: genomeFs.fileSystemId,
      description: "EFS filesystem ID — upload hg38.fa/hg19.fa to /ref before first use",
    });
  }
}
