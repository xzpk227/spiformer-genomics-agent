import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

interface GithubActionsStackProps extends cdk.StackProps {
  // e.g. "your-org/your-repo" or "ryanchang/genomics_project"
  githubRepo: string;
}

export class GithubActionsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: GithubActionsStackProps) {
    super(scope, id, props);

    const { githubRepo } = props;

    const oidcProvider = new iam.OpenIdConnectProvider(this, "GithubOidc", {
      url: "https://token.actions.githubusercontent.com",
      clientIds: ["sts.amazonaws.com"],
      thumbprints: ["6938fd4d98bab03faadb97b34396831e3780aea1"],
    });

    const role = new iam.Role(this, "GithubActionsRole", {
      roleName: "github-actions-deploy",
      assumedBy: new iam.WebIdentityPrincipal(oidcProvider.openIdConnectProviderArn, {
        StringLike: {
          "token.actions.githubusercontent.com:sub": `repo:${githubRepo}:*`,
        },
        StringEquals: {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        },
      }),
    });

    // Scoped to what the pipeline actually needs
    role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ContainerRegistryPowerUser"));
    role.addToPolicy(new iam.PolicyStatement({
      actions: [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:RegisterTaskDefinition",
      ],
      resources: ["*"],
    }));
    role.addToPolicy(new iam.PolicyStatement({
      actions: [
        "cloudformation:*",
        "ssm:GetParameter",
        "sts:AssumeRole",
      ],
      resources: ["*"],
    }));
    role.addToPolicy(new iam.PolicyStatement({
      actions: ["iam:PassRole"],
      resources: ["*"],
      conditions: {
        StringEquals: { "iam:PassedToService": "ecs-tasks.amazonaws.com" },
      },
    }));

    new cdk.CfnOutput(this, "RoleArn", {
      value: role.roleArn,
      description: "Add this as AWS_DEPLOY_ROLE_ARN in GitHub repo secrets",
    });
  }
}
