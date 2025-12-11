# routes/project_routes.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import Session, Project, Dataset, DataItem, audit_logger, logger

project_bp = Blueprint("project_bp", __name__, url_prefix="/api")


@project_bp.route("/projects", methods=["GET"])
@jwt_required()
def list_projects():
    """
    List all projects for the logged-in user.
    """
    user_id = get_jwt_identity()
    session = Session()
    try:
        projects = (
            session.query(Project)
            .filter(Project.owner_id == user_id)
            .order_by(Project.created_at.desc())
            .all()
        )

        result = []
        for p in projects:
            result.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "owner_id": p.owner_id,
                    "quality_tier": p.quality_tier,
                    "schema_version": p.schema_version,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "total_items": p.total_items,
                    "labeled_items": p.labeled_items,
                    "status": p.status,
                }
            )

        return jsonify({"success": True, "projects": result})
    finally:
        session.close()


@project_bp.route("/projects/<project_id>/summary", methods=["GET"])
@jwt_required()
def project_summary(project_id):
    """
    Return a quick summary for one project:
    - counts
    - progress
    """
    user_id = get_jwt_identity()
    session = Session()
    try:
        project = session.query(Project).filter_by(id=project_id).first()
        if not project:
            return jsonify({"error": "Project not found"}), 404

        # OPTIONAL: you could enforce ownership if you want strict access:
        # if project.owner_id != user_id:
        #     return jsonify({"error": "Forbidden"}), 403

        total_items = (
            session.query(DataItem)
            .filter(DataItem.project_id == project_id)
            .count()
        )
        labeled_items = (
            session.query(DataItem)
            .filter(DataItem.project_id == project_id, DataItem.status == "labeled")
            .count()
        )

        progress = (labeled_items / total_items * 100.0) if total_items > 0 else 0.0

        return jsonify(
            {
                "success": True,
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "quality_tier": project.quality_tier,
                },
                "stats": {
                    "total_items": total_items,
                    "labeled_items": labeled_items,
                    "progress_percent": progress,
                },
            }
        )
    finally:
        session.close()
