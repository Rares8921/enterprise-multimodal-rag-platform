variable "project_name" {
  type        = string
  default     = "doc-intel"
  description = "Project name used for resource naming/tagging."
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment (dev/staging/production)."

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be one of: dev, staging, production"
  }
}

variable "primary_region" {
  type        = string
  default     = "us-east-1"
  description = "Primary AWS region."
}

variable "secondary_region" {
  type        = string
  default     = "eu-west-1"
  description = "Secondary AWS region."
}

variable "enable_secondary_region" {
  type        = bool
  default     = true
  description = "Whether to provision the secondary region stack."
}

variable "primary_vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for the primary region VPC."
}

variable "secondary_vpc_cidr" {
  type        = string
  default     = "10.1.0.0/16"
  description = "CIDR block for the secondary region VPC."
}

variable "eks_cluster_version" {
  type        = string
  default     = "1.30"
  description = "EKS cluster Kubernetes version."
}

variable "db_username" {
  type        = string
  default     = "admin"
  description = "RDS master username. Prefer manage_master_user_password=true (default) so no password is stored in code."
}

variable "db_instance_class" {
  type        = string
  default     = "db.r6g.xlarge"
  description = "RDS instance class."
}

variable "db_allocated_storage" {
  type        = number
  default     = 100
  description = "RDS allocated storage (GiB)."
}

variable "redis_node_type" {
  type        = string
  default     = "cache.r6g.xlarge"
  description = "ElastiCache node type."
}

variable "redis_num_cache_clusters" {
  type        = number
  default     = 3
  description = "Number of cache nodes (clusters) in the replication group."
}

variable "redis_auth_token" {
  type        = string
  default     = null
  sensitive   = true
  description = "Optional Redis AUTH token. If set, transit encryption is enabled. Recommended for production."

  validation {
    condition     = var.environment != "production" || (var.redis_auth_token != null && length(var.redis_auth_token) >= 16)
    error_message = "In production you must set redis_auth_token (>= 16 chars) to enable encrypted/authenticated Redis."
  }
}
