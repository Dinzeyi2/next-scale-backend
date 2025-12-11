# routes/admin_routes.py

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import (
    Session,
    User,
    Project,
    DataItem,
    Annotation,
    audit_logger,
    logger,
)

admin_bp = Blueprint("admin_bp", __name__, url_prefix="/api/admin")


def _is_admin(session, user_id: str) -> bool:
    user = session.query(User).filter_by(id=user_id).first()
    return bool(user and user.role == "admin")


@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def list_users():
    """
    Admin-only: list all users.
    """
    current_user_id = get_jwt_identity()
    session = Session()
    try:
        if not _is_admin(session, current_user_id):
            return jsonify({"error": "Admin only"}), 403

        users = session.query(User).order_by(User.created_at.desc()).all()

        result = []
        for u in users:
            result.append(
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "active": u.active,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "last_login": u.last_login.isoformat() if u.last_login else None,
                }
            )

        return jsonify({"success": True, "users": result})
    finally:
        session.close()


@admin_bp.route("/stats", methods=["GET"])
@jwt_required()
def admin_stats():
    """
    Admin-only: global stats.
    """
    current_user_id = get_jwt_identity()
    session = Session()
    try:
        if not _is_admin(session, current_user_id):
            return jsonify({"error": "Admin only"}), 403

        users_count = session.query(User).count()
        projects_count = session.query(Project).count()
        items_count = session.query(DataItem).count()
        annotations_count = session.query(Annotation).count()

        return jsonify(
            {
                "success": True,
                "stats": {
                    "users": users_count,
                    "projects": projects_count,
                    "items": items_count,
                    "annotations": annotations_count,
                },
            }
        )
    finally:
        session.close()
