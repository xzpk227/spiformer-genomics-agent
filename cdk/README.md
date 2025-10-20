# Genomics Agent — CDK Infrastructure

## Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) installed and configured (`aws configure`)
- Node.js 20+ (via `nvm install 20`)
- Docker running locally

## First-time setup

### 1. Install dependencies and bootstrap CDK

```bash
cd cdk
npm install
npx cdk bootstrap
```

### 2. Create secrets in AWS Secrets Manager

```bash
aws secretsmanager create-secret --name genomics/openai-api-key --secret-string "sk-..."
aws secretsmanager create-secret --name genomics/pinecone-api-key --secret-string "pcsk_..."
aws secretsmanager create-secret --name genomics/ncbi-api-key --secret-string "..."
```

### 3. Deploy all stacks

```bash
npx cdk deploy --all
```

### 4. Add GitHub secrets

After deploy, CDK prints two output values. Add them as secrets in your GitHub repo (Settings → Secrets and variables → Actions):

| Secret | Where to find it |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | Output of `GenomicsGithubActionsStack` |
| `ALB_DNS_NAME` | Output of `GenomicsEcsStack` |

Once those are set, pushing to `main` triggers the deploy pipeline automatically.

## Stacks

| Stack | What it creates |
|---|---|
| `GenomicsGithubActionsStack` | OIDC provider + IAM role for GitHub Actions |
| `GenomicsNetworkStack` | VPC, subnets, NAT gateway |
| `GenomicsEcrStack` | ECR repos for backend and frontend |
| `GenomicsStorageStack` | S3 bucket for generated reports (30-day expiry) |
| `GenomicsEcsStack` | Fargate cluster, ALB, backend + frontend services |

## Redeploying after changes

```bash
npx cdk deploy --all        # redeploy all stacks
npx cdk deploy GenomicsEcsStack  # redeploy a single stack
npx cdk diff                # preview changes before deploying
```
