"""create telemetry and predictions tables

Revision ID: 9810c0d5a597
Revises:
Create Date: 2026-11-12 18:51:14.488896

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9810c0d5a597"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("predicted_rul", sa.Float(), nullable=False),
        sa.Column("history_cycles", sa.Integer(), nullable=False),
        sa.Column("is_warmup", sa.Boolean(), nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "unit_id", "cycle", "model_name", name="uq_prediction_unit_cycle_model"
        ),
    )
    op.create_index(
        "ix_predictions_unit_created", "predictions", ["unit_id", "created_at"], unique=False
    )
    op.create_table(
        "telemetry",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("unit_id", sa.Integer(), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("setting_1", sa.Float(), nullable=False),
        sa.Column("setting_2", sa.Float(), nullable=False),
        sa.Column("setting_3", sa.Float(), nullable=False),
        sa.Column("sensor_1", sa.Float(), nullable=False),
        sa.Column("sensor_2", sa.Float(), nullable=False),
        sa.Column("sensor_3", sa.Float(), nullable=False),
        sa.Column("sensor_4", sa.Float(), nullable=False),
        sa.Column("sensor_5", sa.Float(), nullable=False),
        sa.Column("sensor_6", sa.Float(), nullable=False),
        sa.Column("sensor_7", sa.Float(), nullable=False),
        sa.Column("sensor_8", sa.Float(), nullable=False),
        sa.Column("sensor_9", sa.Float(), nullable=False),
        sa.Column("sensor_10", sa.Float(), nullable=False),
        sa.Column("sensor_11", sa.Float(), nullable=False),
        sa.Column("sensor_12", sa.Float(), nullable=False),
        sa.Column("sensor_13", sa.Float(), nullable=False),
        sa.Column("sensor_14", sa.Float(), nullable=False),
        sa.Column("sensor_15", sa.Float(), nullable=False),
        sa.Column("sensor_16", sa.Float(), nullable=False),
        sa.Column("sensor_17", sa.Float(), nullable=False),
        sa.Column("sensor_18", sa.Float(), nullable=False),
        sa.Column("sensor_19", sa.Float(), nullable=False),
        sa.Column("sensor_20", sa.Float(), nullable=False),
        sa.Column("sensor_21", sa.Float(), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("unit_id", "cycle", name="uq_telemetry_unit_cycle"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("telemetry")
    op.drop_index("ix_predictions_unit_created", table_name="predictions")
    op.drop_table("predictions")
