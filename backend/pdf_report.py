import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

logger = logging.getLogger(__name__)

def generate_pdf_report(state: dict, output_path: str):
    """
    Generate a formatted PDF report with dataset stats, models, metrics, and charts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup document
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A1B2F"),
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#555566"),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2C302E"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=10
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#666677"),
        alignment=1,
        spaceBefore=5,
        spaceAfter=15
    )
    
    story = []
    
    # 1. Document Title
    story.append(Paragraph("Autonomous Data Analyst AI Agent", title_style))
    story.append(Paragraph(f"Session Report: {state['session_id']}<br/>ML Analysis & Explainability Summary", subtitle_style))
    story.append(Spacer(1, 15))
    
    # 2. Section: Dataset Profile
    story.append(Paragraph("1. Dataset Profile Summary", h1_style))
    
    profile = state.get("df_profile", {})
    shape = state.get("df_shape", [0, 0])
    
    profile_data = [
        [Paragraph("Metric", bold_body_style), Paragraph("Value", bold_body_style)],
        [Paragraph("Total Records", body_style), Paragraph(str(shape[0]), body_style)],
        [Paragraph("Total Features", body_style), Paragraph(str(shape[1]), body_style)],
        [Paragraph("Target Variable Identified", body_style), Paragraph(str(state.get("target_column") or "N/A (Unsupervised)"), body_style)],
        [Paragraph("Machine Learning Task", body_style), Paragraph(str(state.get("task_type") or "Unknown").capitalize(), body_style)],
    ]
    
    profile_table = Table(profile_data, colWidths=[200, 300])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEEEF5")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 15))
    
    # 3. Section: Correlation & Feature Analysis
    story.append(Paragraph("2. Correlation Analysis", h1_style))
    story.append(Paragraph(
        "A feature correlation matrix was computed to understand collinearity and relations among numeric parameters. Below is the generated Pearson correlation heatmap:", 
        body_style
    ))
    
    charts = state.get("charts", {})
    heatmap_rel_path = charts.get("correlation_heatmap")
    if heatmap_rel_path:
        # Resolve path to disk relative to CWD
        # "/static/charts/xxx.png" -> "static/charts/xxx.png"
        disk_path = heatmap_rel_path.lstrip('/')
        if os.path.exists(disk_path):
            try:
                img = Image(disk_path, width=400, height=300)
                story.append(KeepTogether([img, Paragraph("Figure 1: Pearson Correlation Heatmap matrix of numeric features.", caption_style)]))
            except Exception as e:
                logger.error(f"Failed to add heatmap image to PDF: {e}")
                story.append(Paragraph("[Image loading failed]", caption_style))
        else:
            story.append(Paragraph("[Correlation Heatmap chart not found on disk]", caption_style))
            
    corr_summary = state.get("correlation_summary", [])
    if corr_summary:
        story.append(Paragraph("Top strong correlations identified:", bold_body_style))
        for item in corr_summary:
            txt = f"• <b>{item['feature1']}</b> and <b>{item['feature2']}</b>: Correlation Coefficient = {item['correlation']:.3f}"
            story.append(Paragraph(txt, body_style))
            
    story.append(PageBreak())
    
    # 4. Section: ML Task Detection
    story.append(Paragraph("3. ML Task Detection & Model Selection", h1_style))
    story.append(Paragraph(
        f"<b>Detected Task Type:</b> {str(state.get('task_type')).upper()}", body_style
    ))
    if state.get("reasoning"):
        story.append(Paragraph("<b>Task Detection Decision Logic:</b>", bold_body_style))
        story.append(Paragraph(state["reasoning"], body_style))
    story.append(Spacer(1, 15))
    
    # 5. Section: Model Execution & Metrics
    story.append(Paragraph("4. Model Performance Metrics", h1_style))
    
    ml_results = state.get("ml_results")
    if ml_results:
        metrics = ml_results.get("metrics", {})
        task_type = state.get("task_type")
        
        metrics_data = [[Paragraph("Evaluation Parameter", bold_body_style), Paragraph("Metric Value", bold_body_style)]]
        
        if task_type == "classification":
            metrics_data.extend([
                [Paragraph("Accuracy", body_style), Paragraph(f"{metrics.get('accuracy', 0):.4f}", body_style)],
                [Paragraph("Precision (Weighted)", body_style), Paragraph(f"{metrics.get('precision', 0):.4f}", body_style)],
                [Paragraph("Recall (Weighted)", body_style), Paragraph(f"{metrics.get('recall', 0):.4f}", body_style)],
                [Paragraph("F1-Score (Weighted)", body_style), Paragraph(f"{metrics.get('f1_score', 0):.4f}", body_style)],
            ])
        elif task_type == "clustering":
            metrics_data.extend([
                [Paragraph("Selected Clusters (k)", body_style), Paragraph(str(metrics.get('best_k', 0)), body_style)],
                [Paragraph("Mean Silhouette Score", body_style), Paragraph(f"{metrics.get('silhouette_score', 0):.4f}", body_style)],
            ])
            # Add sizes of clusters
            sizes = ml_results.get("cluster_sizes", {})
            for c_id, count in sizes.items():
                metrics_data.append([Paragraph(f"Cluster {c_id} Size", body_style), Paragraph(f"{count} records", body_style)])
        elif task_type == "anomaly":
            metrics_data.extend([
                [Paragraph("Detected Anomaly Rate", body_style), Paragraph(f"{metrics.get('anomaly_rate', 0) * 100:.2f}%", body_style)],
                [Paragraph("Number of Anomalous Records", body_style), Paragraph(str(metrics.get('anomalies_count', 0)), body_style)],
                [Paragraph("Total Ingested Records", body_style), Paragraph(str(metrics.get('total_records', 0)), body_style)],
                [Paragraph("Mean Isolation Anomaly Score", body_style), Paragraph(f"{metrics.get('mean_anomaly_score', 0):.4f}", body_style)],
            ])
            
        metrics_table = Table(metrics_data, colWidths=[200, 300])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEEEF5")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(metrics_table)
    else:
        story.append(Paragraph("No model execution results available.", body_style))
        
    story.append(Spacer(1, 15))
    
    # 6. Model Visualizations
    story.append(Paragraph("5. Visual Insights & Interpretations", h1_style))
    
    model_chart_key = None
    model_chart_title = ""
    
    if task_type == "classification":
        model_chart_key = "feature_importance"
        model_chart_title = "Feature Attribution / Shapley Feature Importances"
    elif task_type == "clustering":
        model_chart_key = "cluster_scatter"
        model_chart_title = "PCA projection and cluster distribution mapping"
    elif task_type == "anomaly":
        model_chart_key = "anomaly_scatter"
        model_chart_title = "Normal vs. Anomalous distribution mapping"
        
    if model_chart_key:
        chart_rel_path = charts.get(model_chart_key)
        if chart_rel_path:
            disk_path = chart_rel_path.lstrip('/')
            if os.path.exists(disk_path):
                try:
                    img = Image(disk_path, width=400, height=280)
                    story.append(KeepTogether([img, Paragraph(f"Figure 2: {model_chart_title}.", caption_style)]))
                except Exception as e:
                    logger.error(f"Failed to add model chart image to PDF: {e}")
                    story.append(Paragraph("[Image loading failed]", caption_style))
            else:
                story.append(Paragraph("[Model visualization chart not found on disk]", caption_style))
                
    # Add confusion matrix if classification
    if task_type == "classification" and charts.get("confusion_matrix"):
        disk_path = charts["confusion_matrix"].lstrip('/')
        if os.path.exists(disk_path):
            try:
                img = Image(disk_path, width=350, height=290)
                story.append(KeepTogether([img, Paragraph("Figure 3: Classification Confusion Matrix.", caption_style)]))
            except Exception as e:
                logger.error(f"Failed to add confusion matrix image to PDF: {e}")
                
    # Build Document
    doc.build(story)
