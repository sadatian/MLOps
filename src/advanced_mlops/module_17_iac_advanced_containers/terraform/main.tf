terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region                      = var.aws_region
  access_key                  = "mock_access_key"
  secret_key                  = "mock_secret_key"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true

  endpoints {
    s3       = "http://localhost:4566"
    dynamodb = "http://localhost:4566"
  }
}

# S3 Bucket for ML Model Registry
resource "aws_s3_bucket" "model_registry" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Environment = var.environment
    Component   = "MLOps-Registry"
  }
}

# Bucket versioning to support artifact rollbacks
resource "aws_s3_bucket_versioning" "model_registry_versioning" {
  bucket = aws_s3_bucket.model_registry.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server side encryption for compliance / security policy checking
resource "aws_s3_bucket_server_side_encryption_configuration" "model_registry_encryption" {
  bucket = aws_s3_bucket.model_registry.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# DynamoDB Table for ML Model Metadata / serving endpoints
resource "aws_dynamodb_table" "model_metadata" {
  name           = var.table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "ModelId"
  range_key      = "Version"

  attribute {
    name = "ModelId"
    type = "S"
  }

  attribute {
    name = "Version"
    type = "S"
  }

  tags = {
    Environment = var.environment
    Component   = "MLOps-Metadata"
  }
}
