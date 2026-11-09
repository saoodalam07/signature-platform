# Deploy on your own AWS account (low cost)

Design (matches the architecture diagram, scaled down for cost):

```
Internet ──HTTPS──► EC2 (m7i-flex.large, free-plan eligible)
                     ├─ Caddy (HTTPS)  ─► FastAPI
                     ├─ inference worker (RF-DETR + DINOv2, CPU)
                     ├─ Valkey (job results)
                     └─ MLflow (registry)          ◄── laptop registers/promotes models
            Amazon SQS  (queues: inference, training)
            Amazon S3   (temporary uploads, deleted after each job)
Your laptop RTX A3000 = remote GPU trainer (reads the SQS training queue)
Hugging Face dataset push ──webhook──► https://<domain>/webhooks/huggingface
```

New accounts (Free plan) get $100 credit at sign-up (+$100 for small tasks) for 6 months, and cannot be
charged until you upgrade. The Free plan only allows small instances (t3/t4g micro/small,
c7i-flex.large, m7i-flex.large), so we use **m7i-flex.large** (2 vCPU, 8 GB RAM, ≈ $0.10/hour from credits).
**Stop the instance when you are not demoing** to make the credits last.

---

## 1. Safety first: budget alert (2 min)
AWS Console → **Billing → Budgets → Create budget** → *Zero spend budget* (and a $15 monthly budget) → your email.
Setting a budget is also one of the tasks that earns +$20 credit.

## 2. S3 bucket (2 min)
S3 → **Create bucket** → name e.g. `sigplatform-uploads-saood` → region `ap-southeast-2` →
keep **Block all public access** ON → Create.
Bucket → **Management → Lifecycle rule** → expire objects after **1 day** (safety net).

## 3. IAM role for the server (3 min)
IAM → **Policies → Create policy → JSON** → paste `ec2-role-policy.json`
(replace `sigplatform-uploads-CHANGE-ME` with your bucket name) → name `sigplatform-ec2`.
IAM → **Roles → Create role → AWS service → EC2** → attach `sigplatform-ec2` → name `sigplatform-ec2-role`.

## 4. IAM user for the laptop trainer (3 min)
IAM → **Policies → Create policy → JSON** → paste `laptop-trainer-policy.json` → name `sigplatform-trainer`.
IAM → **Users → Create user** `sigplatform-trainer` → attach `sigplatform-trainer`.
User → **Security credentials → Create access key → Local code** → save the two keys
(never paste them in chat or commit them).

## 5. Launch EC2 (5 min)
EC2 → **Launch instance**:
- Name `sigplatform`, AMI **Ubuntu 24.04 LTS**, type **m7i-flex.large** (Free plan eligible)
- Key pair: create `sigplatform-key` (.pem), save it
- Network: create security group with inbound rules
  - SSH 22 → **My IP**
  - HTTP 80 → Anywhere
  - HTTPS 443 → Anywhere
- Storage **40 GB gp3**
- Advanced → **IAM instance profile** → `sigplatform-ec2-role`
- Launch. Optional: **Elastic IP** → allocate → associate, so the address does not change on stop/start.

## 6. Copy the project (Windows CMD, on your laptop)
```
cd C:\Users\DELL\Desktop\signature_project
tar -czf sp.tgz --exclude=venv --exclude=runs_rfdetr --exclude=__pycache__ signature-platform
scp -i C:\path\sigplatform-key.pem sp.tgz ubuntu@<EC2-IP>:~
ssh -i C:\path\sigplatform-key.pem ubuntu@<EC2-IP>
```
(If ssh complains about key permissions: right-click the .pem → Properties → Security → keep only your user.)

## 7. Start the stack (on the server)
```
tar -xzf sp.tgz && cd signature-platform
bash deploy/aws/ec2-setup.sh && newgrp docker
cp deploy/aws/aws.env.example deploy/aws/aws.env && nano deploy/aws/aws.env
```
Set `DOMAIN` (EC2 IP with dashes, e.g. `13-60-1-2.sslip.io`), `S3_BUCKET`, `HF_WEBHOOK_SECRET`. Then:
```
docker compose -f deploy/aws/docker-compose.aws.yml --env-file deploy/aws/aws.env up -d --build
docker compose -f deploy/aws/docker-compose.aws.yml logs -f worker
```

## 8. Put your trained model (v1) into the cloud MLflow (laptop CMD)
MLflow is not open to the internet. Open an SSH tunnel first and keep that window open:
```
ssh -o ServerAliveInterval=30 -i sigplatform-key.pem -N -L 5000:localhost:5000 ubuntu@<EC2-IP>
```
```
cd C:\Users\DELL\Desktop\signature_project\signature-platform
venv\Scripts\activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m training.register_existing --weights runs_rfdetr\signature_rfdetr_large\checkpoint_best_total.pth --map50 0.9649 --map50-95 0.7344 --resolution 576
```

## 9. Test the live API
Open `https://<DOMAIN>/docs` → `GET /v1/models` (v1) → `POST /v1/verify`.

## 10. Laptop as the remote GPU trainer (laptop CMD)
```
set AWS_ACCESS_KEY_ID=<trainer key>
set AWS_SECRET_ACCESS_KEY=<trainer secret>
set AWS_REGION=ap-southeast-2
set AWS_DEFAULT_REGION=ap-southeast-2
set BROKER_URL=sqs://
set RESULT_BACKEND=disabled
set MLFLOW_TRACKING_URI=http://localhost:5000
set HF_TOKEN=<read token>
set TRAIN_WORK_DIR=C:\Users\DELL\Desktop\signature_project\sp_data\work
python -m celery -A sigplatform.queue worker -Q training --pool=solo --loglevel=info
```

## 11. Real Hugging Face webhook
HF → your profile **Settings → Webhooks → Add webhook**:
- Target: dataset repo `saoodalam07/signature-dataset`
- URL `https://<DOMAIN>/webhooks/huggingface`
- Secret = `HF_WEBHOOK_SECRET`, events: **Repo update**

Push new data → HF calls the API → SQS training queue → laptop trains → MLflow → promoted only if
better → cloud worker hot-reloads it (≤ 60 s). Check `GET /v1/models`.

## Stop / start
EC2 → Instance → **Stop** (no compute charge; disk ≈ $3/month). On start, run step 7's `up -d`
(containers restart automatically) and, without an Elastic IP, update `DOMAIN` to the new IP.
