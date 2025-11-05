"""LDAP and Active Directory integration."""

from __future__ import annotations

from typing import List, Optional, Set

try:
    import ldap
    from ldap import LDAPError

    LDAP_AVAILABLE = True
except ImportError:
    LDAP_AVAILABLE = False

    # Create stub LDAPError for when ldap is not available
    class LDAPError(Exception):
        """LDAP error stub."""

        pass


from .models import LDAPConfig, User, AuthProvider


class LDAPHandler:
    """Handle LDAP/Active Directory authentication and user sync."""

    def __init__(self, config: LDAPConfig):
        """Initialize LDAP handler.

        Args:
            config: LDAP configuration

        Raises:
            ImportError: If python-ldap is not installed
        """
        if not LDAP_AVAILABLE:
            raise ImportError(
                "python-ldap is not installed. "
                "Install it with: pip install python-ldap"
            )

        self.config = config
        self._connection: Optional[ldap.ldapobject.LDAPObject] = None

    def connect(self) -> ldap.ldapobject.LDAPObject:
        """Establish LDAP connection.

        Returns:
            LDAP connection object

        Raises:
            LDAPError: If connection fails
        """
        try:
            # Initialize LDAP connection
            if self.config.use_ssl:
                ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)

            conn = ldap.initialize(self.config.server_uri)
            conn.set_option(ldap.OPT_NETWORK_TIMEOUT, self.config.timeout)
            conn.set_option(ldap.OPT_REFERRALS, 0)

            # Bind with service account
            conn.simple_bind_s(self.config.bind_dn, self.config.bind_password)

            self._connection = conn
            return conn

        except LDAPError as e:
            raise LDAPError(f"Failed to connect to LDAP server: {e}") from e

    def disconnect(self):
        """Close LDAP connection."""
        if self._connection:
            try:
                self._connection.unbind_s()
            except LDAPError:
                pass
            finally:
                self._connection = None

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user against LDAP.

        Args:
            username: Username
            password: Password

        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Connect to LDAP
            conn = self.connect()

            # Search for user
            search_filter = self.config.user_search_filter.format(username=username)
            result = conn.search_s(
                self.config.user_search_base,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attrlist=[
                    self.config.username_attribute,
                    self.config.email_attribute,
                    self.config.name_attribute,
                ],
            )

            if not result:
                return None

            user_dn, user_attrs = result[0]

            # Try to bind with user credentials
            try:
                user_conn = ldap.initialize(self.config.server_uri)
                user_conn.simple_bind_s(user_dn, password)
                user_conn.unbind_s()
            except LDAPError:
                return None

            # Get user groups if configured
            groups = []
            if self.config.group_search_base:
                groups = self._get_user_groups(conn, user_dn)

            # Create User object
            user = self._create_user_from_ldap_attrs(
                user_dn,
                user_attrs,
                groups,
            )

            return user

        except LDAPError as e:
            raise LDAPError(f"LDAP authentication failed: {e}") from e
        finally:
            self.disconnect()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user information from LDAP by username.

        Args:
            username: Username

        Returns:
            User object if found, None otherwise
        """
        try:
            conn = self.connect()

            search_filter = self.config.user_search_filter.format(username=username)
            result = conn.search_s(
                self.config.user_search_base,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attrlist=[
                    self.config.username_attribute,
                    self.config.email_attribute,
                    self.config.name_attribute,
                ],
            )

            if not result:
                return None

            user_dn, user_attrs = result[0]

            # Get user groups
            groups = []
            if self.config.group_search_base:
                groups = self._get_user_groups(conn, user_dn)

            user = self._create_user_from_ldap_attrs(user_dn, user_attrs, groups)

            return user

        except LDAPError as e:
            raise LDAPError(f"Failed to get user from LDAP: {e}") from e
        finally:
            self.disconnect()

    def sync_users(self, base_dn: Optional[str] = None) -> List[User]:
        """Sync all users from LDAP.

        Args:
            base_dn: Base DN to search (defaults to user_search_base)

        Returns:
            List of User objects
        """
        try:
            conn = self.connect()

            base = base_dn or self.config.user_search_base
            search_filter = "(objectClass=person)"

            results = conn.search_s(
                base,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attrlist=[
                    self.config.username_attribute,
                    self.config.email_attribute,
                    self.config.name_attribute,
                ],
            )

            users = []
            for user_dn, user_attrs in results:
                if not user_attrs:
                    continue

                groups = []
                if self.config.group_search_base:
                    groups = self._get_user_groups(conn, user_dn)

                user = self._create_user_from_ldap_attrs(user_dn, user_attrs, groups)
                users.append(user)

            return users

        except LDAPError as e:
            raise LDAPError(f"Failed to sync users from LDAP: {e}") from e
        finally:
            self.disconnect()

    def _get_user_groups(self, conn: ldap.ldapobject.LDAPObject, user_dn: str) -> List[str]:
        """Get groups for a user.

        Args:
            conn: LDAP connection
            user_dn: User DN

        Returns:
            List of group names
        """
        if not self.config.group_search_base:
            return []

        try:
            search_filter = self.config.group_search_filter.format(user_dn=user_dn)
            results = conn.search_s(
                self.config.group_search_base,
                ldap.SCOPE_SUBTREE,
                search_filter,
                attrlist=["cn"],
            )

            groups = []
            for _, group_attrs in results:
                if "cn" in group_attrs:
                    groups.extend([g.decode("utf-8") for g in group_attrs["cn"]])

            return groups

        except LDAPError:
            return []

    def _create_user_from_ldap_attrs(
        self,
        user_dn: str,
        user_attrs: dict,
        groups: List[str],
    ) -> User:
        """Create User object from LDAP attributes.

        Args:
            user_dn: User DN
            user_attrs: LDAP attributes
            groups: User groups

        Returns:
            User object
        """
        # Extract attributes
        username_list = user_attrs.get(self.config.username_attribute, [])
        email_list = user_attrs.get(self.config.email_attribute, [])
        name_list = user_attrs.get(self.config.name_attribute, [])

        username = username_list[0].decode("utf-8") if username_list else user_dn
        email = email_list[0].decode("utf-8") if email_list else None
        full_name = name_list[0].decode("utf-8") if name_list else None

        # Map LDAP groups to roles
        roles = self._map_groups_to_roles(groups)

        return User(
            user_id=user_dn,
            username=username,
            email=email,
            full_name=full_name,
            auth_provider=AuthProvider.LDAP,
            roles=roles,
            disabled=False,
        )

    def _map_groups_to_roles(self, groups: List[str]) -> Set[str]:
        """Map LDAP groups to application roles.

        Args:
            groups: LDAP group names

        Returns:
            Set of role names
        """
        # Simple mapping - in production, this would be configurable
        role_mapping = {
            "Domain Admins": "admin",
            "Administrators": "admin",
            "Traders": "trader",
            "Risk Managers": "risk_manager",
            "Analysts": "analyst",
            "Users": "viewer",
        }

        roles = set()
        for group in groups:
            if group in role_mapping:
                roles.add(role_mapping[group])

        # Default role if no mapping found
        if not roles:
            roles.add("viewer")

        return roles

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


__all__ = ["LDAPHandler"]
