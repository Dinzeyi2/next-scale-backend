# routes/export_routes.py

import os
import csv
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import (
    Session,
    Project,
    DataItem,
    Annotation,
    EXPORT_FOLDER,
    audit_logger,
    logger,
)

export_bp = Blueprint("export_bp", __name__, url_prefix="/api")


@export_bp.route("/export/project/<project_id>", methods=["GET"])
@jwt_required()
def export_project_annotations(project_id):
    """
    Export all annotations for a project as CSV.
    Returns a downloadable file.
    """
    user_id = get_jwt_identity()
    session = Session()
    try:
        project = session.query(Project).filter_by(id=project_id).first()
        if not project:
            return jsonify({"error": "Project not found"}), 404

        # Optional ownership check
        # if project.owner_id != user_id:
        #     return jsonify({"error": "Forbidden"}), 403

        # Join DataItem + Annotation
        rows = (
            session.query(DataItem, Annotation)
            .join(Annotation, Annotation.item_id == DataItem.id)
            .filter(DataItem.project_id == project_id)
            .all()
        )

        if not rows:
            return jsonify({"error": "No annotations for this project"}), 400

        # Build CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{project_id}_{timestamp}.csv"
        filepath = os.path.join(EXPORT_FOLDER, filename)

        fieldnames = [
            "project_id",
            "dataset_id",
            "item_id",
            "data_type",
            "file_path",
            "annotator_id",
            "labels_json",
            "codes_json",
            "confidence",
            "time_spent",
            "created_at",
        ]

        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item, ann in rows:
                writer.writerow(
                    {
                        "project_id": item.project_id,
                        "dataset_id": item.dataset_id,
                        "item_id": item.id,
                        "data_type": item.data_type,
                        "file_path": item.file_path,
                        "annotator_id": ann.annotator_id,
                        "labels_json": json.dumps(ann.labels),
                        "codes_json": json.dumps(ann.codes),
                        "confidence": ann.confidence,
                        "time_spent": ann.time_spent,
                        "created_at": ann.created_at.isoformat() if ann.created_at else "",
                    }
                )

        audit_logger.log(
            user_id=user_id,
            action="export_project_annotations",
            resource_type="project",
            resource_id=project_id,
            details={"file": filename},
            ip_address=None,
        )

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype="text/csv",
        )
    finally:
        session.close()
