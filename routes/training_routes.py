# routes/training_routes.py

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from enterprise_complete_system import experiment_controller, audit_logger, logger

training_bp = Blueprint("training_bp", __name__, url_prefix="/api")


@training_bp.route("/train_model", methods=["POST"])
@jwt_required()
def train_model():
    """
    Trigger a simple training / experiment run.
    Body:
      {
        "dataset": "cifar10",
        "num_samples": 10000
      }
    """
    user_id = get_jwt_identity()
    body = request.json or {}
    dataset_name = body.get("dataset", "cifar10")
    num_samples = body.get("num_samples", 10000)

    try:
        results = experiment_controller.run_full_experiment(
            dataset_name=dataset_name,
            num_samples=num_samples,
        )

        audit_logger.log(
            user_id=user_id,
            action="train_model",
            resource_type="experiment",
            resource_id=dataset_name,
            details={"num_samples": num_samples},
            ip_address=None,
        )

        return jsonify({"success": True, "results": results})
    except Exception as e:
        logger.error(f"❌ train_model failed: {e}")
        return jsonify({"error": str(e)}), 500
