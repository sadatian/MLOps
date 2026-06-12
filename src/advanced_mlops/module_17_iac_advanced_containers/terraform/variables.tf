variable "aws_region" {
  description = "AWS region for provisioning resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Name of the S3 bucket used for the model registry"
  type        = string
  default     = "mlops-model-registry-bucket"
}

variable "table_name" {
  description = "Name of the DynamoDB table used for model metadata"
  type        = string
  default     = "mlops-model-metadata-table"
}

variable "environment" {
  description = "Environment tag (e.g. dev, staging, prod)"
  type        = string
  default     = "dev"
}
