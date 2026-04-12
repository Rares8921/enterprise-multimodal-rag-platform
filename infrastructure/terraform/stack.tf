data "aws_caller_identity" "current" {}

data "aws_availability_zones" "primary" {
  state = "available"
}

data "aws_availability_zones" "secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  state    = "available"
  provider = aws.secondary
}

locals {
  is_production = var.environment == "production"

  primary_azs = slice(data.aws_availability_zones.primary.names, 0, 3)
  primary_private_subnets  = [for i in range(3) : cidrsubnet(var.primary_vpc_cidr, 4, i)]
  primary_public_subnets   = [for i in range(3) : cidrsubnet(var.primary_vpc_cidr, 4, i + 3)]
  primary_database_subnets = [for i in range(3) : cidrsubnet(var.primary_vpc_cidr, 4, i + 6)]

  secondary_azs = var.enable_secondary_region ? slice(data.aws_availability_zones.secondary[0].names, 0, 3) : []
  secondary_private_subnets  = var.enable_secondary_region ? [for i in range(3) : cidrsubnet(var.secondary_vpc_cidr, 4, i)] : []
  secondary_public_subnets   = var.enable_secondary_region ? [for i in range(3) : cidrsubnet(var.secondary_vpc_cidr, 4, i + 3)] : []
  secondary_database_subnets = var.enable_secondary_region ? [for i in range(3) : cidrsubnet(var.secondary_vpc_cidr, 4, i + 6)] : []
}

resource "random_id" "primary_bucket" {
  byte_length = 4
  keepers = {
    account_id  = data.aws_caller_identity.current.account_id
    region      = var.primary_region
    project     = var.project_name
    environment = var.environment
  }
}

resource "random_id" "secondary_bucket" {
  count       = var.enable_secondary_region ? 1 : 0
  byte_length = 4
  keepers = {
    account_id  = data.aws_caller_identity.current.account_id
    region      = var.secondary_region
    project     = var.project_name
    environment = var.environment
  }
}

module "vpc_primary" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}"
  cidr = var.primary_vpc_cidr

  azs              = local.primary_azs
  public_subnets   = local.primary_public_subnets
  private_subnets  = local.primary_private_subnets
  database_subnets = local.primary_database_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = local.is_production ? false : true
  one_nat_gateway_per_az = local.is_production ? true : false

  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

module "vpc_secondary" {
  count   = var.enable_secondary_region ? 1 : 0
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  providers = {
    aws = aws.secondary
  }

  name = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}"
  cidr = var.secondary_vpc_cidr

  azs              = local.secondary_azs
  public_subnets   = local.secondary_public_subnets
  private_subnets  = local.secondary_private_subnets
  database_subnets = local.secondary_database_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = local.is_production ? false : true
  one_nat_gateway_per_az = local.is_production ? true : false

  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

module "eks_primary" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-${var.environment}-${var.primary_region}"
  cluster_version = var.eks_cluster_version

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  enable_irsa = true

  vpc_id     = module.vpc_primary.vpc_id
  subnet_ids = module.vpc_primary.private_subnets

  eks_managed_node_groups = {
    general = {
      instance_types = ["t3.xlarge"]
      min_size       = local.is_production ? 2 : 1
      max_size       = local.is_production ? 10 : 3
      desired_size   = local.is_production ? 3 : 1

      labels = {
        workload = "general"
      }
    }

    gpu = {
      instance_types = ["g4dn.xlarge"]
      min_size       = local.is_production ? 1 : 0
      max_size       = local.is_production ? 5 : 1
      desired_size   = local.is_production ? 2 : 0

      labels = {
        workload = "gpu"
      }

      taints = {
        gpu = {
          key    = "gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

module "eks_secondary" {
  count   = var.enable_secondary_region ? 1 : 0
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  providers = {
    aws = aws.secondary
  }

  cluster_name    = "${var.project_name}-${var.environment}-${var.secondary_region}"
  cluster_version = var.eks_cluster_version

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  enable_irsa = true

  vpc_id     = module.vpc_secondary[0].vpc_id
  subnet_ids = module.vpc_secondary[0].private_subnets

  eks_managed_node_groups = {
    general = {
      instance_types = ["t3.xlarge"]
      min_size       = local.is_production ? 2 : 1
      max_size       = local.is_production ? 10 : 3
      desired_size   = local.is_production ? 3 : 1

      labels = {
        workload = "general"
      }
    }

    gpu = {
      instance_types = ["g4dn.xlarge"]
      min_size       = local.is_production ? 1 : 0
      max_size       = local.is_production ? 5 : 1
      desired_size   = local.is_production ? 2 : 0

      labels = {
        workload = "gpu"
      }

      taints = {
        gpu = {
          key    = "gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

# --- S3 (documents) ---
resource "aws_s3_bucket" "documents_primary" {
  bucket        = lower("${var.project_name}-documents-${var.primary_region}-${random_id.primary_bucket.hex}")
  force_destroy = false
}

resource "aws_s3_bucket_ownership_controls" "documents_primary" {
  bucket = aws_s3_bucket.documents_primary.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "documents_primary" {
  bucket = aws_s3_bucket.documents_primary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "documents_primary" {
  bucket = aws_s3_bucket.documents_primary.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents_primary" {
  bucket = aws_s3_bucket.documents_primary.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "documents_primary" {
  bucket = aws_s3_bucket.documents_primary.id

  rule {
    id     = "abort-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "noncurrent-expiry"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "aws_s3_bucket" "documents_secondary" {
  count         = var.enable_secondary_region ? 1 : 0
  provider      = aws.secondary
  bucket        = lower("${var.project_name}-documents-${var.secondary_region}-${random_id.secondary_bucket[0].hex}")
  force_destroy = false
}

resource "aws_s3_bucket_ownership_controls" "documents_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.documents_secondary[0].id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "documents_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.documents_secondary[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "documents_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.documents_secondary[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.documents_secondary[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "documents_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.documents_secondary[0].id

  rule {
    id     = "abort-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "noncurrent-expiry"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# --- RDS ---
resource "aws_security_group" "rds_primary" {
  name_prefix = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}-rds-"
  description = "RDS access from EKS nodes"
  vpc_id      = module.vpc_primary.vpc_id
}

resource "aws_security_group_rule" "rds_primary_from_eks" {
  type                     = "ingress"
  security_group_id        = aws_security_group.rds_primary.id
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.eks_primary.node_security_group_id
}

resource "aws_security_group_rule" "rds_primary_egress" {
  type              = "egress"
  security_group_id = aws_security_group.rds_primary.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

module "rds_primary" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}"

  engine               = "postgres"
  family               = "postgres15"
  major_engine_version = "15"

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage

  db_name  = "doc_intel"
  username = var.db_username

  manage_master_user_password = true

  port = 5432

  vpc_security_group_ids = [aws_security_group.rds_primary.id]

  create_db_subnet_group = true
  subnet_ids             = module.vpc_primary.database_subnets

  multi_az                = local.is_production
  backup_retention_period = local.is_production ? 30 : 7

  storage_encrypted = true

  deletion_protection = local.is_production
  skip_final_snapshot = local.is_production ? false : true
}

resource "aws_security_group" "rds_secondary" {
  count       = var.enable_secondary_region ? 1 : 0
  provider    = aws.secondary
  name_prefix = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}-rds-"
  description = "RDS access from EKS nodes"
  vpc_id      = module.vpc_secondary[0].vpc_id
}

resource "aws_security_group_rule" "rds_secondary_from_eks" {
  count                    = var.enable_secondary_region ? 1 : 0
  provider                 = aws.secondary
  type                     = "ingress"
  security_group_id        = aws_security_group.rds_secondary[0].id
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.eks_secondary[0].node_security_group_id
}

resource "aws_security_group_rule" "rds_secondary_egress" {
  count             = var.enable_secondary_region ? 1 : 0
  provider          = aws.secondary
  type              = "egress"
  security_group_id = aws_security_group.rds_secondary[0].id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

module "rds_secondary" {
  count   = var.enable_secondary_region ? 1 : 0
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  providers = {
    aws = aws.secondary
  }

  identifier = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}"

  engine               = "postgres"
  family               = "postgres15"
  major_engine_version = "15"

  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage

  db_name  = "doc_intel"
  username = var.db_username

  manage_master_user_password = true

  port = 5432

  vpc_security_group_ids = [aws_security_group.rds_secondary[0].id]

  create_db_subnet_group = true
  subnet_ids             = module.vpc_secondary[0].database_subnets

  multi_az                = local.is_production
  backup_retention_period = local.is_production ? 30 : 7

  storage_encrypted = true

  deletion_protection = local.is_production
  skip_final_snapshot = local.is_production ? false : true
}

# --- Redis ---
resource "aws_security_group" "redis_primary" {
  name_prefix = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}-redis-"
  description = "Redis access from EKS nodes"
  vpc_id      = module.vpc_primary.vpc_id
}

resource "aws_security_group_rule" "redis_primary_from_eks" {
  type                     = "ingress"
  security_group_id        = aws_security_group.redis_primary.id
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  source_security_group_id = module.eks_primary.node_security_group_id
}

resource "aws_security_group_rule" "redis_primary_egress" {
  type              = "egress"
  security_group_id = aws_security_group.redis_primary.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_elasticache_subnet_group" "redis_primary" {
  name       = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}-redis"
  subnet_ids = module.vpc_primary.private_subnets
}

resource "aws_elasticache_replication_group" "redis_primary" {
  replication_group_id = "${var.project_name}-${var.environment}-${replace(var.primary_region, "-", "")}"
  description          = "${var.project_name} Redis"

  engine         = "redis"
  engine_version = "7.1"

  node_type          = var.redis_node_type
  num_cache_clusters = var.redis_num_cache_clusters
  port               = 6379

  subnet_group_name  = aws_elasticache_subnet_group.redis_primary.name
  security_group_ids = [aws_security_group.redis_primary.id]

  automatic_failover_enabled = var.redis_num_cache_clusters > 1
  multi_az_enabled           = var.redis_num_cache_clusters > 1

  at_rest_encryption_enabled = true

  transit_encryption_enabled = var.redis_auth_token != null
  auth_token                 = var.redis_auth_token

  maintenance_window = "sun:03:00-sun:04:00"

  apply_immediately = !local.is_production
}

resource "aws_security_group" "redis_secondary" {
  count       = var.enable_secondary_region ? 1 : 0
  provider    = aws.secondary
  name_prefix = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}-redis-"
  description = "Redis access from EKS nodes"
  vpc_id      = module.vpc_secondary[0].vpc_id
}

resource "aws_security_group_rule" "redis_secondary_from_eks" {
  count                    = var.enable_secondary_region ? 1 : 0
  provider                 = aws.secondary
  type                     = "ingress"
  security_group_id        = aws_security_group.redis_secondary[0].id
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  source_security_group_id = module.eks_secondary[0].node_security_group_id
}

resource "aws_security_group_rule" "redis_secondary_egress" {
  count             = var.enable_secondary_region ? 1 : 0
  provider          = aws.secondary
  type              = "egress"
  security_group_id = aws_security_group.redis_secondary[0].id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_elasticache_subnet_group" "redis_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary
  name     = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}-redis"

  subnet_ids = module.vpc_secondary[0].private_subnets
}

resource "aws_elasticache_replication_group" "redis_secondary" {
  count    = var.enable_secondary_region ? 1 : 0
  provider = aws.secondary

  replication_group_id = "${var.project_name}-${var.environment}-${replace(var.secondary_region, "-", "")}"
  description          = "${var.project_name} Redis"

  engine         = "redis"
  engine_version = "7.1"

  node_type          = var.redis_node_type
  num_cache_clusters = var.redis_num_cache_clusters
  port               = 6379

  subnet_group_name  = aws_elasticache_subnet_group.redis_secondary[0].name
  security_group_ids = [aws_security_group.redis_secondary[0].id]

  automatic_failover_enabled = var.redis_num_cache_clusters > 1
  multi_az_enabled           = var.redis_num_cache_clusters > 1

  at_rest_encryption_enabled = true

  transit_encryption_enabled = var.redis_auth_token != null
  auth_token                 = var.redis_auth_token

  maintenance_window = "sun:03:00-sun:04:00"

  apply_immediately = !local.is_production
}
