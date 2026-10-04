resource "docker_network" "mlops_net" {
  name = "mlops_tf_net"
}


resource "docker_image" "postgres" {
  name         = "postgres:14-alpine"
  keep_locally = true
}

resource "docker_container" "postgres" {
  image = docker_image.postgres.image_id
  name  = "postgres-tf"
  env = [
    "POSTGRES_USER=${var.db_user}",
    "POSTGRES_PASSWORD=${var.db_password}",
    "POSTGRES_DB=${var.db_name}"
  ]
  ports {
    internal = 5432
    external = 5432
  }
  networks_advanced {
    name = docker_network.mlops_net.name
  }
}


resource "docker_image" "minio" {
  name         = "quay.io/minio/minio"
  keep_locally = true
}

resource "docker_container" "minio" {
  image   = docker_image.minio.image_id
  name    = "minio-tf"
  command = ["server", "/data", "--console-address", ":9001"]
  env = [
    "MINIO_ROOT_USER=${var.minio_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_password}"
  ]
  ports {
    internal = 9000
    external = 9000
  }
  ports {
    internal = 9001
    external = 9001
  }
  networks_advanced {
    name = docker_network.mlops_net.name
  }
}


resource "docker_image" "mlflow" {
  name         = "python:3.11-slim"
  keep_locally = true
}

resource "docker_container" "mlflow" {
  image = docker_image.mlflow.image_id
  name  = "mlflow-tf"
  command = [
    "bash", "-c",
    "pip install mlflow psycopg2-binary boto3 && mlflow server --backend-store-uri postgresql://${var.db_user}:${var.db_password}@postgres-tf:5432/${var.db_name} --default-artifact-root s3://mlflow/ --host 0.0.0.0"
  ]
  env = [
    "AWS_ACCESS_KEY_ID=${var.minio_user}",
    "AWS_SECRET_ACCESS_KEY=${var.minio_password}",
    "MLFLOW_S3_ENDPOINT_URL=http://minio-tf:9000"
  ]
  ports {
    internal = 5000
    external = 5000
  }
  networks_advanced {
    name = docker_network.mlops_net.name
  }
}
