"""Role based access control helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set

from .audit import AuditLogger
from .tenancy import TenantManager


@dataclass(slots=True)
class Role:
    """Role definition with a permission set."""

    name: str
    permissions: Set[str] = field(default_factory=set)
    description: str = ""

    def allows(self, permission: str) -> bool:
        return permission in self.permissions


class RBACManager:
    """Manage roles and assignments with optional tenant scoping."""

    def __init__(
        self,
        *,
        tenant_manager: TenantManager | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self._tenant_manager = tenant_manager
        self._audit = audit_logger
        self._lock = RLock()
        self._global_roles: Dict[str, Role] = {}
        self._tenant_roles: Dict[str, Dict[str, Role]] = {}
        self._global_assignments: Dict[str, Set[str]] = {}
        self._tenant_assignments: Dict[str, Dict[str, Set[str]]] = {}

    def define_role(
        self,
        role: Role,
        *,
        tenant_id: str | None = None,
        overwrite: bool = False,
    ) -> Role:
        """Register a role globally or within a tenant."""

        target = self._roles_for_tenant(tenant_id)
        with self._lock:
            exists = role.name in target
            if exists and not overwrite:
                scope = tenant_id or "global"
                raise ValueError(f"Role '{role.name}' already exists in scope '{scope}'")
            target[role.name] = Role(role.name, set(role.permissions), role.description)
        self._log(
            action="rbac.role.define",
            tenant_id=tenant_id,
            metadata={"role": role.name, "overwrite": overwrite},
        )
        return target[role.name]

    def update_role_permissions(
        self,
        role_name: str,
        permissions: Iterable[str],
        *,
        tenant_id: str | None = None,
    ) -> Role:
        """Replace the permission set for a role."""

        target = self._roles_for_tenant(tenant_id)
        with self._lock:
            if role_name not in target:
                scope = tenant_id or "global"
                raise KeyError(f"Role '{role_name}' is not defined in scope '{scope}'")
            target[role_name].permissions = set(permissions)
            role = target[role_name]
        self._log(
            action="rbac.role.permissions.update",
            tenant_id=tenant_id,
            metadata={"role": role_name, "permissions": list(permissions)},
        )
        return role

    def assign_role(
        self,
        user_id: str,
        role_name: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Assign ``role_name`` to ``user_id``."""

        role = self._get_role(role_name, tenant_id=tenant_id)
        if tenant_id and self._tenant_manager is not None:
            self._tenant_manager.ensure(tenant_id)
        with self._lock:
            assignments = self._assignments_for_tenant(tenant_id)
            user_roles = assignments.setdefault(user_id, set())
            user_roles.add(role.name)
        self._log(
            action="rbac.role.assign",
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={"role": role_name},
        )

    def revoke_role(
        self,
        user_id: str,
        role_name: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Remove a role assignment from a user."""

        with self._lock:
            assignments = self._assignments_for_tenant(tenant_id)
            user_roles = assignments.get(user_id)
            if not user_roles or role_name not in user_roles:
                return
            user_roles.remove(role_name)
            if not user_roles:
                assignments.pop(user_id, None)
        self._log(
            action="rbac.role.revoke",
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={"role": role_name},
        )

    def check_access(
        self,
        user_id: str,
        permission: str,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        """Return ``True`` if the user has the required permission."""

        for role_name in self.get_roles_for_user(user_id, tenant_id=tenant_id):
            role = self._resolve_role(role_name, tenant_id)
            if role.allows(permission):
                return True
        if tenant_id is not None:
            # fallback to global assignments if tenant-specific check failed
            for role_name in self.get_roles_for_user(user_id):
                role = self._global_roles.get(role_name)
                if role and role.allows(permission):
                    return True
        return False

    def get_roles_for_user(self, user_id: str, *, tenant_id: str | None = None) -> List[str]:
        """Return roles assigned to ``user_id``."""

        assignments = self._assignments_for_tenant(tenant_id)
        with self._lock:
            roles = list(assignments.get(user_id, ()))
        return roles

    def get_effective_permissions(
        self,
        user_id: str,
        *,
        tenant_id: str | None = None,
    ) -> Set[str]:
        """Return aggregate permissions for ``user_id``."""

        permissions: Set[str] = set()
        for role_name in self.get_roles_for_user(user_id, tenant_id=tenant_id):
            role = self._resolve_role(role_name, tenant_id)
            permissions.update(role.permissions)
        if tenant_id is not None:
            for role_name in self.get_roles_for_user(user_id):
                role = self._global_roles.get(role_name)
                if role:
                    permissions.update(role.permissions)
        return permissions

    def assignments(self, *, tenant_id: str | None = None) -> Dict[str, Set[str]]:
        """Return a copy of role assignments in the requested scope."""

        assignments = self._assignments_for_tenant(tenant_id)
        with self._lock:
            return {user: set(roles) for user, roles in assignments.items()}

    def all_tenant_assignments(self) -> Dict[str, Dict[str, Set[str]]]:
        """Return tenant-scoped assignments."""

        with self._lock:
            return {
                tenant: {user: set(roles) for user, roles in assignments.items()}
                for tenant, assignments in self._tenant_assignments.items()
            }

    def has_role(self, role_name: str, *, tenant_id: str | None = None) -> bool:
        """Return ``True`` if the role exists in the requested scope."""

        with self._lock:
            if tenant_id is None:
                return role_name in self._global_roles
            return role_name in self._tenant_roles.get(tenant_id, {})

    def _roles_for_tenant(self, tenant_id: str | None) -> Dict[str, Role]:
        if tenant_id is None:
            return self._global_roles
        with self._lock:
            return self._tenant_roles.setdefault(tenant_id, {})

    def _assignments_for_tenant(self, tenant_id: str | None) -> Dict[str, Set[str]]:
        if tenant_id is None:
            return self._global_assignments
        with self._lock:
            return self._tenant_assignments.setdefault(tenant_id, {})

    def _has_role(self, role_name: str, tenant_id: str | None) -> bool:
        roles = self._roles_for_tenant(tenant_id)
        with self._lock:
            return role_name in roles

    def _get_role(self, role_name: str, *, tenant_id: str | None = None) -> Role:
        roles = self._roles_for_tenant(tenant_id)
        with self._lock:
            role = roles.get(role_name)
        if role:
            return role
        if tenant_id is None:
            raise KeyError(f"Role '{role_name}' is not defined")
        with self._lock:
            role = self._global_roles.get(role_name)
        if role is None:
            raise KeyError(f"Role '{role_name}' is not defined in scope '{tenant_id}' or globally")
        return role

    def _resolve_role(self, role_name: str, tenant_id: str | None) -> Role:
        if tenant_id is not None and self._has_role(role_name, tenant_id):
            return self._get_role(role_name, tenant_id=tenant_id)
        return self._get_role(role_name)

    def _log(
        self,
        *,
        action: str,
        tenant_id: str | None = None,
        user_id: str | None = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if self._audit is not None:
            payload = dict(metadata or {})
            self._audit.log(action=action, tenant_id=tenant_id, user_id=user_id, metadata=payload)


__all__ = ["RBACManager", "Role"]
