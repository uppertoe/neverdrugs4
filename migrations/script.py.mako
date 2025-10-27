##
## Alembic migration script template.
##
from __future__ import annotations

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

from alembic import op
import sqlalchemy as sa  # noqa: F401


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
