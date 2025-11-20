-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create meeting_insights table
CREATE TABLE IF NOT EXISTS meeting_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    enterprise_id UUID NOT NULL,
    product_id TEXT,
    metric_key TEXT NOT NULL,
    details_json JSONB,
    embedding vector(1024), -- Dimension inferred from settings
    cluster_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create insight_clusters table
CREATE TABLE IF NOT EXISTS insight_clusters (
    id TEXT PRIMARY KEY,
    enterprise_id UUID NOT NULL,
    product_id TEXT,
    metric_key TEXT NOT NULL,
    centroid vector(1024), -- Dimension inferred from settings
    size INTEGER DEFAULT 0,
    name TEXT,
    description TEXT,
    type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes (optional but recommended)
CREATE INDEX IF NOT EXISTS idx_meeting_insights_metric_key ON meeting_insights(metric_key);
CREATE INDEX IF NOT EXISTS idx_meeting_insights_cluster_id ON meeting_insights(cluster_id);
CREATE INDEX IF NOT EXISTS idx_insight_clusters_enterprise_metric ON insight_clusters(enterprise_id, metric_key);
