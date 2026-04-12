/*
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

variable "project_name" {
  default = "doc-intel"
}

variable "regions" {
  type = list(string)
  default = ["us-east-1", "eu-west-1"]
}

variable "environment" {
  default = "production"
}

provider "aws" {
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu-west-1"
  region = "eu-west-1"
}

locals {
  providers = {
    "us-east-1" = aws
    "eu-west-1" = aws.eu-west-1
  }
}

module "vpc" {
  source = "./modules/vpc"

  for_each = toset(var.regions)

  providers = {
    aws = local.providers[each.value]
  }

  region = each.value
  project_name = var.project_name
  cidr_block = "10.${index(var.regions, each.value)}.0.0/16"
}

module "eks" {
  source = "./modules/eks"

  for_each = toset(var.regions)

  providers = {
    aws = local.providers[each.value]
  }

  region = each.value
  project_name = var.project_name
  vpc_id = module.vpc[each.value].vpc_id
  subnet_ids = module.vpc[each.value].private_subnet_ids

  node_groups = {
    general = {
      desired_size  = 3
      min_size = 2
      max_size = 10
      instance_types = ["t3.xlarge"]
      labels = {
        workload = "general"
      }
    }
    gpu = {
      desired_size = 2
      min_size = 1
      max_size = 5
      instance_types = ["g4dn.xlarge"]
      labels = {
        workload = "gpu"
      }
      taints = [{
        key = "gpu"
        value = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

module "rds" {
  source = "./modules/rds"

  for_each = toset(var.regions)

  providers = {
    aws = local.providers[each.value]
  }

  region = each.value
  project_name = var.project_name
  vpc_id = module.vpc[each.value].vpc_id
  subnet_ids = module.vpc[each.value].database_subnet_ids

  instance_class = "db.r6g.xlarge"
  allocated_storage = 100
  multi_az = true
  backup_retention_period = 30
  backup_window = "03:00-04:00"
}

module "redis" {
  source = "./modules/redis"

  for_each = toset(var.regions)

  providers = {
    aws = local.providers[each.value]
  }

  region = each.value
  project_name = var.project_name
  vpc_id = module.vpc[each.value].vpc_id
  subnet_ids = module.vpc[each.value].private_subnet_ids

  node_type = "cache.r6g.xlarge"
  num_cache_nodes = 3
  automatic_failover_enabled = true
}

resource "aws_s3_bucket" "documents" {
  for_each = toset(var.regions)

  provider = local.providers[each.value]

  bucket = "${var.project_name}-documents-${each.value}"

  tags = {
    Name = "${var.project_name}-documents"
    Region = each.value
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  for_each = toset(var.regions)

  provider = local.providers[each.value]

  bucket = aws_s3_bucket.documents[each.value].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_route53_zone" "main" {
  name = "doc-intel.example.com"
}

resource "aws_route53_record" "api" {
  for_each = toset(var.regions)

  zone_id = aws_route53_zone.main.zone_id
  name = "api-${each.value}.doc-intel.example.com"
  type = "A"

  alias {
    name = module.eks[each.value].load_balancer_hostname
    zone_id = module.eks[each.value].load_balancer_zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api_global" {
  for_each = toset(var.regions)

  zone_id = aws_route53_zone.main.zone_id
  name = "api.doc-intel.example.com"
  type = "A"

  set_identifier = each.value

  latency_routing_policy {
    region = each.value
  }

  alias {
    name = aws_route53_record.api[each.value].name
    zone_id = aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }
}

module "monitoring" {
  source = "./modules/monitoring"

  for_each = toset(var.regions)

  providers = {
    aws = local.providers[each.value]
  }

  region = each.value
  project_name = var.project_name
  eks_cluster_name  = module.eks[each.value].cluster_name
}

output "api_endpoints" {
  value = {
    for region in var.regions :
    region => "https://api-${region}.doc-intel.example.com"
  }
}

output "eks_cluster_names" {
  value = {
    for region in var.regions :
    region => module.eks[region].cluster_name
  }
}

output "database_endpoints" {
  value = {
    for region in var.regions :
    region => module.rds[region].endpoint
  }
  sensitive = true
}
*/