output "registry_bucket_arn" {
  description = "The ARN of the model registry S3 bucket"
  value       = aws_s3_bucket.model_registry.arn
}

output "metadata_table_name" {
  description = "The name of the model metadata DynamoDB table"
  value       = aws_dynamodb_table.model_metadata.name
}
