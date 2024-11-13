%%sh
docker build . -f Dockerfile.inference --platform linux/amd64 -t prithvi_global_inference

export ECR_URL="637423382292.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password --region us-west-2 | \
  docker login --password-stdin --username AWS $ECR_URL

docker tag prithvi_global_inference $ECR_URL/prithvi_global_inference:latest

docker push $ECR_URL/prithvi_global_inference:latest
