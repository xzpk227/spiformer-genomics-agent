import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as logs from "aws-cdk-lib/aws-logs";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

interface EcsStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  backendRepo: ecr.Repository;
  frontendRepo: ecr.Repository;
  reportsBucket: s3.Bucket;
}

export class EcsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: EcsStackProps) {
    super(scope, id, props);

    const { vpc, backendRepo, frontendRepo, reportsBucket } = props;

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
      healthCheck: {
        command: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
      },
    });

    const backendSg = new ec2.SecurityGroup(this, "BackendSg", { vpc });
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
        elbv2.ListenerCondition.pathPatterns(["/chat", "/health", "/reports/*"]),
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
  }
}
