# Neutryx Kubernetes Deployment Guide

## Overview

This guide covers deploying Neutryx on Kubernetes with auto-scaling, multi-region support, and disaster recovery capabilities.

## Prerequisites

- Kubernetes 1.25+
- kubectl CLI configured
- Helm 3.x
- Cloud provider CLI (gcloud, aws, or az)
- cert-manager (for TLS certificates)
- Prometheus Operator (for monitoring)

## Quick Start

### 1. Deploy with Kubectl

```bash
# Create namespace
kubectl apply -f k8s/base/namespace.yaml

# Deploy base resources
kubectl apply -k k8s/

# Verify deployment
kubectl get pods -n neutryx
kubectl get svc -n neutryx
```

### 2. Deploy with Helm

```bash
# Add Helm repository (if using external chart)
helm repo add neutryx https://charts.neutryx.io
helm repo update

# Install with default values
helm install neutryx k8s/helm/neutryx \
  --namespace neutryx \
  --create-namespace

# Install with custom values
helm install neutryx k8s/helm/neutryx \
  --namespace neutryx \
  --create-namespace \
  --values my-values.yaml
```

### 3. Deploy with Kustomize

```bash
# Development environment
kubectl apply -k k8s/overlays/dev

# Staging environment
kubectl apply -k k8s/overlays/staging

# Production environment
kubectl apply -k k8s/overlays/prod
```

## Architecture

### Components

1. **API Servers** (`neutryx-api`)
   - REST API endpoints
   - Auto-scaling: 3-50 pods
   - Resource limits: 2 CPU, 4Gi RAM per pod

2. **Workers** (`neutryx-worker`)
   - Background processing
   - Auto-scaling: 5-100 pods
   - Resource limits: 4 CPU, 8Gi RAM per pod

3. **Redis** (`neutryx-redis`)
   - Cache and queue backend
   - StatefulSet with persistent storage
   - 3 replicas for high availability

4. **Monitoring Stack**
   - Prometheus for metrics
   - Grafana for visualization
   - Alert Manager for notifications

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

#### API Autoscaling
```yaml
spec:
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          averageUtilization: 80
```

#### Worker Autoscaling
```yaml
spec:
  minReplicas: 5
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          averageUtilization: 75
    - type: Pods
      pods:
        metric:
          name: queue_depth
        target:
          averageValue: "100"
```

### Scale-up/Scale-down Behavior

**API Pods:**
- Scale up: +100% or +4 pods every 30s (whichever is higher)
- Scale down: -50% or -2 pods every 60s (whichever is lower)
- Stabilization window: 60s up, 300s down

**Worker Pods:**
- Scale up: +200% or +10 pods every 30s (whichever is higher)
- Scale down: -25% or -5 pods every 120s (whichever is lower)
- Stabilization window: 30s up, 600s down

### Custom Metrics

To use custom metrics (e.g., queue depth, request rate):

```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Install Prometheus Adapter for custom metrics
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --values prometheus-adapter-values.yaml
```

## Multi-Region Deployment

### Architecture

Neutryx supports active-active multi-region deployment across:
- **US Central** (primary)
- **Europe West** (secondary)
- **Asia Southeast** (tertiary)

### Global Load Balancing

The global load balancer distributes traffic based on:
1. Geographic proximity (latency-based routing)
2. Health of regional clusters
3. Capacity and current load

### Setup

1. **Deploy to Each Region:**

```bash
# US Central
gcloud container clusters create neutryx-us \
  --region=us-central1 \
  --num-nodes=5 \
  --machine-type=n2-standard-8

kubectl apply -k k8s/overlays/prod --context=us-central

# Europe West
gcloud container clusters create neutryx-eu \
  --region=europe-west1 \
  --num-nodes=5 \
  --machine-type=n2-standard-8

kubectl apply -k k8s/overlays/prod --context=europe-west

# Asia Southeast
gcloud container clusters create neutryx-asia \
  --region=asia-southeast1 \
  --num-nodes=5 \
  --machine-type=n2-standard-8

kubectl apply -k k8s/overlays/prod --context=asia-southeast
```

2. **Configure Global Load Balancer:**

```bash
kubectl apply -f k8s/multi-region/global-load-balancer.yaml
```

3. **Verify Multi-Region Setup:**

```bash
# Check health of all regions
for region in us-central europe-west asia-southeast; do
  echo "=== $region ==="
  kubectl get pods -n neutryx --context=$region
done
```

### Traffic Routing

Traffic is routed using:
- **Geo-based routing**: Users routed to nearest region
- **Failover**: Automatic failover to backup region on failure
- **Sticky sessions**: Session affinity for stateful operations

## Disaster Recovery

### Backup Strategy

#### Automated Backups

1. **Database Backups** (every 6 hours)
   ```bash
   kubectl apply -f k8s/disaster-recovery/backup-cronjob.yaml
   ```

2. **State Backups** (daily)
   - Redis snapshots
   - Application state
   - Configuration

#### Retention Policy

- **Hourly**: 24 backups (1 day)
- **Daily**: 30 backups (1 month)
- **Weekly**: 12 backups (3 months)
- **Monthly**: 12 backups (1 year)

### Recovery Procedures

#### Scenario 1: Single Pod Failure
- **Automatic**: Kubernetes restarts pod
- **RTO**: ~30 seconds
- **RPO**: 0 (no data loss)

#### Scenario 2: Node Failure
- **Automatic**: Pods rescheduled to healthy nodes
- **RTO**: 2-3 minutes
- **RPO**: 0

#### Scenario 3: Zone Failure
- **Automatic**: Traffic routed to other zones
- **RTO**: 1-2 minutes
- **RPO**: 0

#### Scenario 4: Region Failure
- **Automatic**: Global LB routes to backup region
- **Manual**: Scale up backup region if needed
- **RTO**: 5-10 minutes
- **RPO**: 0 (with multi-region replication)

#### Scenario 5: Data Corruption
- **Manual**: Restore from backup
- **RTO**: 15-30 minutes
- **RPO**: Up to 6 hours (last backup)

**Steps:**
```bash
# List available backups
gsutil ls gs://neutryx-backups/database/

# Restore specific backup
export BACKUP_FILE="neutryx_backup_20240101_120000.sql.gz"
kubectl create -f k8s/disaster-recovery/restore-job.yaml \
  --dry-run=client -o yaml | \
  sed "s|RESTORE_BACKUP_FILE: \"\"|RESTORE_BACKUP_FILE: \"$BACKUP_FILE\"|" | \
  kubectl apply -f -

# Monitor restoration
kubectl logs -f job/neutryx-db-restore -n neutryx
```

### Testing DR Procedures

Monthly DR drill checklist:
- [ ] Verify backups are running
- [ ] Test backup restoration in dev
- [ ] Simulate region failure
- [ ] Verify auto-scaling behavior
- [ ] Test manual failover procedures
- [ ] Update runbooks if needed

## Monitoring and Alerts

### Prometheus Metrics

Key metrics exposed:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `pricing_calculations_total`: Pricing calculations
- `queue_depth`: Worker queue depth
- `database_connections`: Active DB connections

### Grafana Dashboards

Pre-built dashboards available:
1. **Overview**: System health and key metrics
2. **API Performance**: Request rates, latency, errors
3. **Worker Performance**: Queue depth, processing time
4. **Resource Usage**: CPU, memory, disk
5. **Business Metrics**: Calculations, trades, pricing

Access Grafana:
```bash
kubectl port-forward svc/grafana 3000:80 -n monitoring
# Open http://localhost:3000
```

### Alert Configuration

Critical alerts configured for:
- API/Worker downtime (>1 minute)
- High error rates (>5%)
- High latency (p95 >1 second)
- Queue backlog (>10,000 items)
- Resource exhaustion (>90% CPU/memory)
- Database connection failures
- Backup failures

Configure alert notifications:
```bash
kubectl edit configmap alertmanager-config -n monitoring
```

## Security

### Network Policies

```bash
kubectl apply -f k8s/security/network-policies.yaml
```

### Pod Security Policies

- Non-root user execution
- Read-only root filesystem
- Drop all capabilities
- No privilege escalation

### Secrets Management

Use Kubernetes Secrets or external secret managers:

```bash
# Create secrets
kubectl create secret generic neutryx-secrets \
  --from-literal=db-password='...' \
  --from-literal=redis-password='...' \
  --from-literal=api-key='...' \
  -n neutryx

# Or use external secrets operator
kubectl apply -f k8s/security/external-secrets.yaml
```

### TLS/SSL Configuration

Managed by cert-manager:
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create cluster issuer
kubectl apply -f k8s/security/cluster-issuer.yaml
```

## Scaling Recommendations

### Node Sizing

**Development**:
- 3 nodes × n2-standard-4 (4 vCPU, 16GB RAM)

**Staging**:
- 5 nodes × n2-standard-8 (8 vCPU, 32GB RAM)

**Production**:
- 10-50 nodes × n2-standard-16 (16 vCPU, 64GB RAM)
- Node auto-scaling enabled
- Mixed instance types for cost optimization

### Storage Classes

```yaml
fast-ssd:
  type: pd-ssd  # GCP
  # type: gp3  # AWS
  iops: 10000
  throughput: 500MB/s

standard:
  type: pd-standard  # GCP
  # type: gp2  # AWS
```

## Cost Optimization

1. **Auto-scaling**: Pods scale down during low traffic
2. **Node auto-scaling**: Cluster scales with demand
3. **Spot/Preemptible instances**: Use for worker nodes
4. **Resource requests**: Set appropriate requests/limits
5. **Storage lifecycle**: Archive old backups to cheaper storage

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n neutryx
kubectl describe pod <pod-name> -n neutryx

# Check logs
kubectl logs <pod-name> -n neutryx
kubectl logs <pod-name> -n neutryx --previous
```

### High CPU/Memory Usage

```bash
# Check resource usage
kubectl top pods -n neutryx
kubectl top nodes

# Scale manually if needed
kubectl scale deployment neutryx-api --replicas=10 -n neutryx
```

### Service Not Accessible

```bash
# Check service and endpoints
kubectl get svc -n neutryx
kubectl get endpoints -n neutryx

# Check ingress
kubectl get ingress -n neutryx
kubectl describe ingress neutryx-ingress -n neutryx
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h <db-host> -U <db-user> -d <db-name>

# Check secrets
kubectl get secret neutryx-db-secret -n neutryx -o yaml
```

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/neutryx-api \
  neutryx-api=neutryx/neutryx-core:v1.1.0 \
  -n neutryx

# Monitor rollout
kubectl rollout status deployment/neutryx-api -n neutryx

# Rollback if needed
kubectl rollout undo deployment/neutryx-api -n neutryx
```

### Backup Verification

```bash
# List recent backups
gsutil ls -lh gs://neutryx-backups/database/ | tail -10

# Download and verify backup
gsutil cp gs://neutryx-backups/database/latest.sql.gz /tmp/
gunzip -t /tmp/latest.sql.gz
```

## Support

- **Documentation**: https://docs.neutryx.io
- **Issues**: https://github.com/neutryx-lab/neutryx-core/issues
- **Email**: support@neutryx.io
- **Emergency**: See k8s/disaster-recovery/dr-procedures.md
