# routes/annotation_routes.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import (
    Session,
    DataItem,
    Annotation,
    Project,
    audit_logger,
    logger,
)

annotation_bp = Blueprint("annotation_bp", __name__, url_prefix="/api")


@annotation_bp.route("/annotations", methods=["POST"])
@jwt_required()
def submit_annotation():
    """
    Submit an annotation for a given item.

    Expected JSON body:
    {
      "item_id": "...",
      "labels": {...},
      "codes": {...},
      "confidence": 0.9,
      "time_spent": 3.5
    }
    """
    user_id = get_jwt_identity()
    data = request.json or {}

    item_id = data.get("item_id")
    labels = data.get("labels")
    codes = data.get("codes")
    confidence = data.get("confidence", 1.0)
    time_spent = data.get("time_spent", 0.0)

    if not item_id or labels is None or codes is None:
        return jsonify({"error": "item_id, labels and codes are required"}), 400

    session = Session()
    try:
        item = session.query(DataItem).filter_by(id=item_id).first()
        if not item:
            return jsonify({"error": "Item not found"}), 404

        ann = Annotation(
            id=str(__import__("uuid").uuid4()),
            item_id=item.id,
            annotator_id=user_id,
            labels=labels,
            codes=codes,
            confidence=float(confidence),
            time_spent=float(time_spent),
        )
        session.add(ann)

        # Mark item as labeled
        item.status = "labeled"

        # Update project counters
        project = session.query(Project).filter_by(id=item.project_id).first()
        if project:
            project.labeled_items = project.labeled_items + 1

        session.commit()

        audit_logger.log(
            user_id=user_id,
            action="submit_annotation",
            resource_type="annotation",
            resource_id=ann.id,
            details={"item_id": item_id},
            ip_address=None,
        )

        return jsonify({"success": True, "annotation_id": ann.id})
    finally:
        session.close()


@annotation_bp.route("/annotations", methods=["GET"])
@jwt_required()
def list_annotations():
    """
    List annotations for a project (or dataset/item).

    Query params:
      - project_id (optional)
      - item_id (optional)
    """
    user_id = get_jwt_identity()
    project_id = request.args.get("project_id")
    item_id = request.args.get("item_id")

    session = Session()
    try:
        query = session.query(Annotation)

        if item_id:
            query = query.filter(Annotation.item_id == item_id)
        elif project_id:
            # Join via DataItem
            from sqlalchemy.orm import aliased
            di_alias = aliased(DataItem)
            query = (
                session.query(Annotation)
                .join(di_alias, Annotation.item_id == di_alias.id)
                .filter(di_alias.project_id == project_id)
            )

        annotations = query.order_by(Annotation.created_at.desc()).limit(500).all()

        result = []
        for a in annotations:
            result.append(
                {
                    "id": a.id,
                    "item_id": a.item_id,
                    "annotator_id": a.annotator_id,
                    "labels": a.labels,
                    "codes": a.codes,
                    "confidence": a.confidence,
                    "time_spent": a.time_spent,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
            )

        return jsonify({"success": True, "annotations": result})
    finally:
        session.close()
