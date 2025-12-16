import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export class EcrStack extends cdk.Stack {
  readonly backendRepo: ecr.Repository;
  readonly frontendRepo: ecr.Repository;
  readonly spliformerRepo: ecr.Repository;

  constructor(scope: Construct, id: string, props: cdk.StackProps) {
    super(scope, id, props);

    this.backendRepo = new ecr.Repository(this, "BackendRepo", {
      repositoryName: "genomics-backend",
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [{ maxImageCount: 10 }],
    });

    this.frontendRepo = new ecr.Repository(this, "FrontendRepo", {
      repositoryName: "genomics-frontend",
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [{ maxImageCount: 10 }],
    });

    this.spliformerRepo = new ecr.Repository(this, "SpliformerRepo", {
      repositoryName: "genomics-spliformer",
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [{ maxImageCount: 10 }],
    });

    new cdk.CfnOutput(this, "BackendRepoUri", { value: this.backendRepo.repositoryUri });
    new cdk.CfnOutput(this, "FrontendRepoUri", { value: this.frontendRepo.repositoryUri });
    new cdk.CfnOutput(this, "SpliformerRepoUri", { value: this.spliformerRepo.repositoryUri });
  }
}
