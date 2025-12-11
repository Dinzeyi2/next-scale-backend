# routes/item_routes.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import (
    Session,
    DataItem,
    Dataset,
    Project,
    storage_manager,
    audit_logger,
    logger,
)

item_bp = Blueprint("item_bp", __name__, url_prefix="/api")


@item_bp.route("/items/next", methods=["GET"])
@jwt_required()
def get_next_item():
    """
    Get the next pending item for annotation for a project.
    Query params:
      - project_id (required)
      - data_type (optional: image/video/text/audio)
    """
    user_id = get_jwt_identity()
    project_id = request.args.get("project_id")
    data_type = request.args.get("data_type")

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    session = Session()
    try:
        query = session.query(DataItem).filter(
            DataItem.project_id == project_id,
            DataItem.status == "pending",
        )

        if data_type:
            query = query.filter(DataItem.data_type == data_type)

        item = query.order_by(DataItem.created_at.asc()).first()

        if not item:
            return jsonify({"success": True, "item": None, "message": "No pending items"}), 200

        # We only return the path; frontend can decide how to use it
        # (e.g., direct URL if you later use S3 public URLs)
        response_item = {
            "id": item.id,
            "dataset_id": item.dataset_id,
            "project_id": item.project_id,
            "data_type": item.data_type,
            "file_path": item.file_path,
            "status": item.status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

        # Optionally mark as "in_progress"
        item.status = "in_progress"
        session.commit()

        audit_logger.log(
            user_id=user_id,
            action="fetch_next_item",
            resource_type="data_item",
            resource_id=item.id,
            details={"project_id": project_id, "data_type": data_type},
            ip_address=None,
        )

        return jsonify({"success": True, "item": response_item})
    finally:
        session.close()
