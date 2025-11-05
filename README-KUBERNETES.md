# Neutryx Kubernetes Deployment

Enterprise-grade Kubernetes deployment for Neutryx quantitative finance platform with auto-scaling, multi-region support, and disaster recovery.

## Features

✅ **Auto-Scaling**
- Horizontal Pod Autoscaler (HPA) with CPU, memory, and custom metrics
- API: 3-50 pods, Workers: 5-100 pods
- Intelligent scale-up/scale-down policies
- Pod Disruption Budgets for high availability

✅ **Multi-Region Deployment**
- Active-active across 3 regions (US, EU, Asia)
- Global load balancing with latency-based routing
- Automatic failover and traffic shifting
- Geographic redundancy

✅ **Disaster Recovery**
- Automated backups every 6 hours
- Multi-tier retention (hourly, daily, weekly, monthly)
- RTO: 15 minutes, RPO: 6 hours
- Comprehensive runbooks and procedures

✅ **Monitoring & Observability**
- Prometheus metrics and alerting
- Grafana dashboards
- Distributed tracing support
- Business and infrastructure metrics

✅ **Security**
- Network policies and Pod Security Policies
- TLS/SSL with cert-manager
- Secrets management
- Non-root containers

## Quick Start

### Option 1: Kubectl (Production-ready)

```bash
# Deploy to production
kubectl apply -k k8s/

# Verify
kubectl get pods -n neutryx
kubectl get svc -n neutryx
kubectl get hpa -n neutryx
```

### Option 2: Helm (Recommended)

```bash
# Install
helm install neutryx k8s/helm/neutryx \
  --namespace neutryx \
  --create-namespace \
  --values my-values.yaml

# Upgrade
helm upgrade neutryx k8s/helm/neutryx \
  --namespace neutryx \
  --values my-values.yaml

# Status
helm status neutryx -n neutryx
```

### Option 3: Kustomize (Environment-specific)

```bash
# Development
kubectl apply -k k8s/overlays/dev

# Staging
kubectl apply -k k8s/overlays/staging

# Production
kubectl apply -k k8s/overlays/prod
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Global Load Balancer                         │
│              (Latency-based + Health-check routing)              │
└────────┬───────────────────────┬─────────────────────┬──────────┘
         │                       │                     │
    ┌────▼─────┐           ┌────▼─────┐         ┌────▼─────┐
    │ US-Central│           │ EU-West  │         │ Asia-SE  │
    │  Region   │           │  Region  │         │  Region  │
    └────┬─────┘           └────┬─────┘         └────┬─────┘
         │                       │                     │
    ┌────▼──────────────────────▼─────────────────────▼────┐
    │              Kubernetes Clusters                      │
    │  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  │
    │  │ API Pods   │  │Worker Pods  │  │ Redis        │  │
    │  │ (3-50)     │  │ (5-100)     │  │ StatefulSet  │  │
    │  │  + HPA     │  │  + HPA      │  │ (3 replicas) │  │
    │  └────────────┘  └─────────────┘  └──────────────┘  │
    └───────────────────────────────────────────────────────┘
```

## Documentation

- **[Full Deployment Guide](docs/deployment/KUBERNETES.md)** - Complete setup instructions
- **[Disaster Recovery](k8s/disaster-recovery/restore-job.yaml)** - DR procedures
- **[Monitoring Setup](k8s/monitoring/prometheus-rules.yaml)** - Alerts and dashboards
- **[Helm Chart](k8s/helm/neutryx/)** - Helm deployment options

## Configuration

### Environment Variables

Key configuration options (see `k8s/base/configmap.yaml`):

```yaml
LOG_LEVEL: INFO
API_WORKERS: 4
WORKER_CONCURRENCY: 4
ENABLE_METRICS: true
ENABLE_TRACING: true
```

### Resource Limits

**API Pods:**
- Request: 500m CPU, 1Gi RAM
- Limit: 2 CPU, 4Gi RAM

**Worker Pods:**
- Request: 1 CPU, 2Gi RAM
- Limit: 4 CPU, 8Gi RAM

### Auto-Scaling Targets

**API:** 70% CPU, 80% Memory, 1000 req/s
**Workers:** 75% CPU, 85% Memory, 100 queue depth

## Monitoring

Access monitoring dashboards:

```bash
# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Grafana
kubectl port-forward svc/grafana 3000:80 -n monitoring
```

### Key Metrics

- `http_requests_total` - API request count
- `http_request_duration_seconds` - Request latency
- `pricing_calculations_total` - Business metrics
- `celery_queue_length` - Worker queue depth

### Alerts

Critical alerts configured:
- API/Worker downtime
- High error rates (>5%)
- High latency (p95 >1s)
- Resource exhaustion
- Queue backlog
- Database failures

## Disaster Recovery

### Backup Schedule

- **Database**: Every 6 hours
- **State**: Daily
- **Retention**: 24h/30d/12w/12m

### Recovery Procedures

```bash
# List backups
gsutil ls gs://neutryx-backups/database/

# Restore from backup
kubectl apply -f k8s/disaster-recovery/restore-job.yaml

# Monitor restoration
kubectl logs -f job/neutryx-db-restore -n neutryx
```

### Failover

**Automatic:** Regional failure triggers global LB redirect
**Manual:** Scale backup region, update DNS if needed

## Production Checklist

Before going to production:

- [ ] Configure secrets (DB, Redis, API keys)
- [ ] Set up TLS certificates
- [ ] Configure monitoring alerts
- [ ] Test backup/restore procedures
- [ ] Configure multi-region if needed
- [ ] Set resource requests/limits
- [ ] Enable network policies
- [ ] Configure pod security policies
- [ ] Set up log aggregation
- [ ] Test auto-scaling behavior
- [ ] Configure external DNS
- [ ] Set up CI/CD pipelines

## Scaling

### Manual Scaling

```bash
# Scale API
kubectl scale deployment neutryx-api --replicas=20 -n neutryx

# Scale Workers
kubectl scale deployment neutryx-worker --replicas=50 -n neutryx
```

### Cluster Scaling

```bash
# Add nodes
gcloud container clusters resize neutryx --num-nodes=20 --region=us-central1

# Enable node auto-scaling
gcloud container clusters update neutryx \
  --enable-autoscaling \
  --min-nodes=10 \
  --max-nodes=50 \
  --region=us-central1
```

## Troubleshooting

### Common Issues

**Pods not starting:**
```bash
kubectl describe pod <pod-name> -n neutryx
kubectl logs <pod-name> -n neutryx
```

**High latency:**
```bash
kubectl top pods -n neutryx
kubectl get hpa -n neutryx
```

**Connection issues:**
```bash
kubectl get svc -n neutryx
kubectl get ingress -n neutryx
```

See [Full Guide](docs/deployment/KUBERNETES.md#troubleshooting) for more.

## Support

- **Documentation**: https://docs.neutryx.io
- **GitHub**: https://github.com/neutryx-lab/neutryx-core
- **Email**: support@neutryx.io

## License

Apache 2.0 - See LICENSE file for details
