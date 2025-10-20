#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { NetworkStack } from "../lib/network-stack";
import { EcrStack } from "../lib/ecr-stack";
import { StorageStack } from "../lib/storage-stack";
import { EcsStack } from "../lib/ecs-stack";
import { GithubActionsStack } from "../lib/github-actions-stack";

const app = new cdk.App();

const githubRepo = "xzpk227/genomics-agent";

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
};

new GithubActionsStack(app, "GenomicsGithubActionsStack", { env, githubRepo });

const network = new NetworkStack(app, "GenomicsNetworkStack", { env });
const ecr = new EcrStack(app, "GenomicsEcrStack", { env });
const storage = new StorageStack(app, "GenomicsStorageStack", { env, vpc: network.vpc });

new EcsStack(app, "GenomicsEcsStack", {
  env,
  vpc: network.vpc,
  backendRepo: ecr.backendRepo,
  frontendRepo: ecr.frontendRepo,
  reportsBucket: storage.reportsBucket,
});
