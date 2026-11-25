/**
 * Standalone MCP client for Raptor — no LLM, no agent framework.
 *
 * Takes an already-obtained JWT (e.g. from Raptor's `POST /api/0.4/sso/login`,
 * or `examples/curl_mcp.sh` which can log in for you) and drives the MCP
 * protocol directly over Streamable HTTP: initialize, list tools, call a
 * tool, read a resource.
 *
 * Requires: npm install @modelcontextprotocol/sdk tsx
 *
 * Usage:
 *   npx tsx examples/typescript_client.ts --jwt <token>
 *   npx tsx examples/typescript_client.ts --jwt <token> --server-url http://localhost:8027/mcp
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

function parseArgs(argv: string[]): { jwt: string; serverUrl: string } {
  let jwt: string | undefined;
  let serverUrl = "http://localhost:8027/mcp";
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--jwt") jwt = argv[++i];
    else if (argv[i] === "--server-url") serverUrl = argv[++i] ?? serverUrl;
  }
  if (!jwt) {
    console.error("Usage: typescript_client.ts --jwt <token> [--server-url <url>]");
    process.exit(1);
  }
  return { jwt, serverUrl };
}

function textOf(result: object): string {
  if (!("content" in result)) return "";
  return ((result.content ?? []) as Array<{ type: string; text?: string }>)
    .filter((b) => b.type === "text" && b.text)
    .map((b) => b.text)
    .join("");
}

async function main() {
  const { jwt, serverUrl } = parseArgs(process.argv.slice(2));

  const transport = new StreamableHTTPClientTransport(new URL(serverUrl), {
    requestInit: { headers: { Authorization: `Bearer ${jwt}` } },
  });

  const client = new Client({ name: "raptor-direct-client", version: "1.0.0" });

  await client.connect(transport);
  const serverInfo = client.getServerVersion();
  console.log(`Connected to ${serverInfo?.name} v${serverInfo?.version}\n`);

  const tools = await client.listTools();
  console.log(`Available tools (${tools.tools.length}):`);
  for (const tool of tools.tools) console.log(`  - ${tool.name}`);
  console.log();

  const searchResult = await client.callTool({
    name: "raptor_search",
    arguments: { query: "video", top_k: 3 },
  });
  const parsed = JSON.parse(textOf(searchResult));
  if (!Array.isArray(parsed)) {
    throw new Error(`raptor_search failed: ${JSON.stringify(parsed)}`);
  }
  const results = parsed as Array<{ score?: number; asset_path?: string }>;
  console.log(`raptor_search('video') -> ${results.length} hit(s):`);
  for (const hit of results) {
    console.log(`  - [${(hit.score ?? 0).toFixed(3)}] ${hit.asset_path ?? ""}`);
  }
  console.log();
  const resource = await client.readResource({ uri: "raptor://capabilities" });
  const capabilitiesMd = (resource.contents[0] as { text: string }).text;
  console.log(`raptor://capabilities (${capabilitiesMd.length} chars):`);
  console.log(capabilitiesMd.slice(0, 300) + "...");

  await client.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
