# routes/dataset_routes.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import Session, Dataset, Project, audit_logger, logger

dataset_bp = Blueprint("dataset_bp", __name__, url_prefix="/api")


@dataset_bp.route("/datasets", methods=["GET"])
@jwt_required()
def list_datasets():
    """
    List datasets for a given project_id.
    Query param: ?project_id=...
    """
    user_id = get_jwt_identity()
    project_id = request.args.get("project_id")

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    session = Session()
    try:
        project = session.query(Project).filter_by(id=project_id).first()
        if not project:
            return jsonify({"error": "Project not found"}), 404

        # Optional ownership check:
        # if project.owner_id != user_id:
        #     return jsonify({"error": "Forbidden"}), 403

        datasets = (
            session.query(Dataset)
            .filter(Dataset.project_id == project_id)
            .order_by(Dataset.uploaded_at.desc())
            .all()
        )

        result = []
        for d in datasets:
            result.append(
                {
                    "id": d.id,
                    "name": d.name,
                    "data_type": d.data_type,
                    "num_files": d.num_files,
                    "size_bytes": d.size_bytes,
                    "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                    "processed": d.processed,
                }
            )

        return jsonify({"success": True, "datasets": result})
    finally:
        session.close()
