# Deploy on your Linux VPS

This guide delivers deployable configuration; it does not imply that a server has already been connected or deployed. The reference commands target **Ubuntu 24.04**, with Python 3.12, systemd and Nginx. On another distribution, adjust package installation and provide Python 3.12 first.

The production flow is:

```text
HTTPS → Nginx → 127.0.0.1:8000 Gunicorn → Flask + local Chroma → Gemini API
```

Run one Gunicorn process with four threads and at most two simultaneous questions. This keeps the embedded index on one machine and leaves capacity for health requests. No Docker, Node service, Redis, or hosted vector database is needed.

## 1. Prepare the machine and upload this checkout

On the VPS, install the runtime:

```sh
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv nginx rsync
sudo useradd --system --home-dir /var/lib/tldrracket --shell /usr/sbin/nologin tldrracket
sudo install -d -o root -g root -m 755 /opt/tldrracket/releases
sudo install -d -o tldrracket -g tldrracket -m 700 /var/lib/tldrracket
```

If the `tldrracket` account already exists, reuse it instead of running `useradd` again. Point your domain's DNS A record at the VPS; add an AAAA record only if the server has working IPv6. Allow SSH and ports 80/443 in your existing firewall. Port 8000 stays on loopback.

From your local repository, upload the reviewed code and intended public readings. Replace `deployuser@your-server` with your SSH login or configured host alias:

```sh
rsync -av --exclude='.git/' --exclude='.venv/' --include='.env.example' --exclude='.env*' --exclude='chroma/' --exclude='__pycache__/' ./ deployuser@your-server:/tmp/tldrracket-upload/
```

This copies your actual checkout, including uncommitted implementation files; it does not assume the remote Git repository already contains these changes. Keep real credentials out of the upload directory. For subsequent releases, use a new upload/release directory.

On the VPS, install this first release:

```sh
sudo cp -a /tmp/tldrracket-upload /opt/tldrracket/releases/2026-10-22-1
sudo chown -R root:root /opt/tldrracket/releases/2026-10-22-1
sudo chmod -R a+rX /opt/tldrracket/releases/2026-10-22-1
sudo python3.12 -m venv /opt/tldrracket/releases/2026-10-22-1/.venv
sudo /opt/tldrracket/releases/2026-10-22-1/.venv/bin/python -m pip install -r /opt/tldrracket/releases/2026-10-22-1/requirements.txt
sudo ln -s /opt/tldrracket/releases/2026-10-22-1 /opt/tldrracket/current
```

Create each virtual environment at its final release path, since its executable wrappers contain absolute paths. Code and dependencies remain owned by root; the running service can write only its state directory. If `current` already exists, follow the upgrade procedure below.

## 2. Configure the API and persistent index

```sh
sudo install -o root -g tldrracket -m 640 /opt/tldrracket/current/.env.example /etc/tldrracket.env
sudoedit /etc/tldrracket.env
```

Set your real key, retain the model settings unless intentionally migrating, and change these paths:

```dotenv
GOOGLE_API_KEY=your-real-key
DATA_PATH=/opt/tldrracket/current/data
CHROMA_PATH=/var/lib/tldrracket/chroma
MAX_DISTANCE=0.35
```

Use shell-compatible `KEY=value` entries because the manual maintenance commands source this file. Do not use `export` prefixes; quote values containing spaces. The service and maintenance commands must share this same file. The initial distance cutoff is provisional; calibrate it before treating it as tuned.

Build the index as the service account. This calls the embedding API, not the chat model:

```sh
sudo -u tldrracket /bin/sh -c 'set -a; . /etc/tldrracket.env; set +a; cd /opt/tldrracket/current; exec .venv/bin/python index.py build'
```

Optionally inspect and tune retrieval using actual corpus embeddings:

```sh
sudo -u tldrracket /bin/sh -c 'set -a; . /etc/tldrracket.env; set +a; cd /opt/tldrracket/current; exec .venv/bin/python index.py probe "How do closures capture an environment?"'
sudo -u tldrracket /bin/sh -c 'set -a; . /etc/tldrracket.env; set +a; cd /opt/tldrracket/current; exec .venv/bin/python index.py calibrate docs/retrieval-eval.json'
```

Review the seed labels, add held-out questions, and choose `MAX_DISTANCE` from the observed behavior. The cutoff is a distance, with lower values meaning a closer match.

## 3. Start Gunicorn under systemd

```sh
sudo install -m 644 /opt/tldrracket/current/deploy/tldrracket.service /etc/systemd/system/tldrracket.service
sudo systemd-analyze verify /etc/systemd/system/tldrracket.service
sudo systemctl daemon-reload
sudo systemctl enable --now tldrracket
sudo systemctl status tldrracket --no-pager
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
```

`/healthz` checks the process. `/readyz` opens and validates a compatible non-empty index and checks that the API key is configured, without calling Gemini. It cannot establish whether the remote key, model or quota works. Make a real question request after readiness passes:

```sh
curl --fail-with-body http://127.0.0.1:8000/api/ask -H 'Content-Type: application/json' -d '{"message":"What is lexical scope?"}'
```

Check the answer, source preview, and linked PDF page. Ask an unrelated question as well, and confirm the chosen threshold does not force five sources.

## 4. Add Nginx, domain and HTTPS

```sh
sudo install -m 644 /opt/tldrracket/current/deploy/nginx.conf /etc/nginx/sites-available/tldrracket
sudoedit /etc/nginx/sites-available/tldrracket
```

Replace `readings.example.com` with your actual domain. Install the file from Nginx's **http context** as shown below; its `limit_req_zone` directive must not be nested inside another server block.

```sh
sudo ln -s /etc/nginx/sites-available/tldrracket /etc/nginx/sites-enabled/tldrracket
sudo nginx -t
sudo systemctl reload nginx
```

The sample limits `/api/ask` to six requests per minute per client IP with a small burst and returns HTTP 429 on excess. Adjust it for your intended traffic. Requests forwarded through another proxy require an explicit trusted real-IP configuration before relying on per-IP limits.

Obtain a certificate using your existing certificate manager or [Certbot's current Nginx installation instructions](https://certbot.eff.org/instructions). Once Certbot is installed, the issuance and renewal checks are:

```sh
sudo certbot --nginx -d readings.example.com
sudo certbot renew --dry-run
```

Use your actual domain in the command. Select the HTTPS redirect when prompted. Verify the configured renewal timer and open your site's HTTPS URL. Test a course question, an unrelated question, a reading link and the waiting state in a browser. Keep the certificate-managed Nginx configuration when later upgrading the application.

## Operations

View logs:

```sh
sudo journalctl -u tldrracket -n 100 --no-pager
sudo journalctl -u tldrracket -f
```

The browser waits at most 90 seconds. Nginx's upstream read timeout is 95 seconds. The provider transport timeout defaults to 30 seconds for embedding and generation independently, with no automatic retries. Gunicorn's 120-second timeout governs worker health; with a threaded worker it is not a hard per-request cancellation deadline. Browser cancellation does not guarantee cancellation of an already-running upstream request.

### Update readings or embedding settings

Stop the service, then back up the **entire** Chroma directory and the corresponding readings/configuration. Do not copy an actively written SQLite/HNSW directory. Store the environment-file backup securely because it contains the API key.

```sh
sudo systemctl stop tldrracket
sudo cp -a /var/lib/tldrracket/chroma /var/lib/tldrracket/chroma.backup-2026-10-22
```

Update the reading files in a new release. For normal material changes, use `index.py build`. For an embedding model/dimension/chunk migration, an old notebook index, or an index built before Markdown whitespace preservation (schema version 1), use the explicit replacement command:

```sh
sudo -u tldrracket /bin/sh -c 'set -a; . /etc/tldrracket.env; set +a; cd /opt/tldrracket/current; exec .venv/bin/python index.py build --rebuild'
sudo systemctl start tldrracket
curl --fail http://127.0.0.1:8000/readyz
```

`--rebuild` replaces only the `tldrracket` collection and does not delete unrelated collections. Recheck your cutoff after changing embeddings or the corpus.

### Recover an interrupted build

A failed or interrupted write keeps `CHROMA_PATH/.building`, and the service refuses to answer from that partial index. First stop the web service and verify that no indexing process is still running:

```sh
sudo systemctl stop tldrracket
pgrep -af 'python.*index.py'
```

Only after the previous indexing process has exited, remove the marker and run `index.py build --rebuild` with the same environment file. At the default VPS path, marker removal is:

```sh
sudo -u tldrracket rm /var/lib/tldrracket/chroma/.building
sudo -u tldrracket /bin/sh -c 'set -a; . /etc/tldrracket.env; set +a; cd /opt/tldrracket/current; exec .venv/bin/python index.py build --rebuild'
sudo systemctl start tldrracket
curl --fail http://127.0.0.1:8000/readyz
```

Alternatively restore the stopped-service backup together with its matching code, readings and embedding settings. Do not remove the marker merely to make readiness green: it indicates an incomplete write.

### Upgrade and roll back

Upload each new release to its own directory under `/opt/tldrracket/releases/`, create its virtual environment there, install dependencies, and run its offline tests before switching. Use a new directory name instead of overwriting the running release. Save the previous release path and back up state/configuration before changes.

With the service stopped, update the `current` symlink to the prepared release. For example, after preparing `/opt/tldrracket/releases/2026-10-22-2`:

```sh
sudo systemctl stop tldrracket
sudo ln -sfn /opt/tldrracket/releases/2026-10-22-2 /opt/tldrracket/current
```

Run `index.py build` from the new release if the corpus changed, or `build --rebuild` if embedding/chunk settings changed. Restart the service and verify readiness plus real questions. A cutoff or chat-model-only change needs a restart but no re-embedding.

For rollback, stop the service, point `current` back to the previous release, restore the matching index/readings/environment backup if a migration occurred, then start and verify again. A code-only rollback can reuse a compatible index. Keep `/var/lib/tldrracket` outside release directories so upgrades never erase it.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `/healthz` fails | systemd status/journal, virtualenv path, permissions, port binding |
| `/readyz` returns 503 | key presence, manifest, `.building`, empty/incompatible index, file ownership |
| Question returns 502/504 | server journal, available model, API key/quota/network, provider timeout |
| Too many irrelevant sources | inspect `probe`, lower/calibrate `MAX_DISTANCE`, review corpus and labels |
| Useful course questions return no match | inspect candidates; verify indexing, model compatibility and cutoff |
| PDF link fails | file remains under `DATA_PATH`, source filenames match, page is the PDF page index |
| 429 response | two active questions or Nginx request rate limit; wait before retrying |

The systemd and Nginx files are examples to validate on the target VPS using the commands above. Local application tests cannot validate a server's DNS, certificates, systemd version, or network setup.
