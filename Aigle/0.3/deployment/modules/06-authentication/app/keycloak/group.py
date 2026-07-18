import requests

from app.core.config import settings


class KeycloakGroup:
    @staticmethod
    def creat_groups_api(group_name: str, realm: str, token: str):
        payload = {"name": group_name}
        return requests.post(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/groups",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    @staticmethod
    def get_groups_api(realm: str, token: str):
        return requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/groups",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

    @staticmethod
    def resolve_group_id(realm_name: str, group_path: str, token: str) -> str:
        """Resolves group path like '/parent/child' to Keycloak UUID.
        Raises ValueError if not found, RuntimeError on API error."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        parts = [p for p in group_path.split("/") if p]
        if not parts:
            raise ValueError(f"Invalid group path: '{group_path}'")

        group_id = None
        found = None
        for part in parts:
            if group_id is None:
                url = f"{settings.KEYCLOAK_URL}/admin/realms/{realm_name}/groups"
            else:
                url = f"{settings.KEYCLOAK_URL}/admin/realms/{realm_name}/groups/{group_id}/children"

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"Keycloak API error resolving '{group_path}': {response.text}")

            found = next((g for g in response.json() if g.get("name") == part), None)
            if not found:
                raise ValueError(f"Group not found: '{group_path}' (failed at '{part}')")
            group_id = found["id"]

        return group_id

    @staticmethod
    def get_user_groups_api(user_id: str, realm: str, token: str):
        """GET /admin/realms/{realm}/users/{userId}/groups — 回傳使用者所屬的所有 group 物件。"""
        return requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/groups",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
