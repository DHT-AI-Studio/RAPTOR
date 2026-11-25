"""SeaweedFS storage for session recordings."""
from __future__ import annotations

import httpx


class BlobError(Exception):
    pass


class SeaweedFS:
    def __init__(self, master_url: str, client: httpx.AsyncClient, collection: str = "", timeout_s: float = 30.0) -> None:
        self.master = master_url.rstrip("/")
        self.client = client
        self.collection = collection
        self.timeout_s = timeout_s

    async def put(self, data: bytes, name: str, mime: str = "application/octet-stream") -> str:
        params = {"collection": self.collection} if self.collection else {}
        try:
            assign = (await self.client.get(f"{self.master}/dir/assign", params=params, timeout=self.timeout_s)).json()
            if "fid" not in assign:
                raise BlobError(f"assign failed: {assign.get('error', assign)}")
            resp = await self.client.post(f"http://{assign['url']}/{assign['fid']}",
                                          files={"file": (name, data, mime)}, timeout=self.timeout_s)
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise BlobError(str(exc)) from exc
        return assign["fid"]

    async def _url(self, fid: str) -> str | None:
        vid = fid.split(",", 1)[0]
        resp = await self.client.get(f"{self.master}/dir/lookup", params={"volumeId": vid}, timeout=self.timeout_s)
        resp.raise_for_status()
        locations = resp.json().get("locations") or []
        return f"http://{locations[0]['url']}/{fid}" if locations else None

    async def get(self, fid: str) -> bytes:
        try:
            url = await self._url(fid)
            if url is None:
                raise BlobError(f"{fid}: volume not found")
            resp = await self.client.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.content
        except (httpx.HTTPError, ValueError) as exc:
            raise BlobError(str(exc)) from exc

    async def delete(self, fid: str) -> int:
        """Bytes freed; 0 when the blob is already gone."""
        try:
            url = await self._url(fid)
            if url is None:
                raise BlobError(f"{fid}: volume not found; deletion unconfirmed")
            resp = await self.client.delete(url, timeout=self.timeout_s)
            if resp.status_code == 404:
                return 0
            resp.raise_for_status()
            return int(resp.json().get("size", 0))
        except (httpx.HTTPError, ValueError) as exc:
            raise BlobError(str(exc)) from exc
