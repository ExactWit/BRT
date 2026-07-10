#!/usr/bin/env bash
# Sync repo-root models.tsv (exp_launcher) -> scripts/model_registry.tsv (branch.sh).
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_DIR}/models.tsv"
DST="${REPO_DIR}/scripts/model_registry.tsv"

if [[ ! -f "${SRC}" ]]; then
  echo "[sync_model_registry] missing ${SRC}" >&2
  exit 1
fi

{
  printf 'id\tlabel\tgit_ref\tbranch\trun_tag\tstatus\tmetrics_doc\n'
  tail -n +2 "${SRC}" | while IFS=$'\t' read -r id label repo_name git_ref branch run_tag status notes _; do
    [[ -z "${id}" || "${id}" == "id" ]] && continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${id}" "${label}" "${git_ref}" "${branch}" "${run_tag}" "${status}" "${notes}"
  done
} > "${DST}.tmp"
mv "${DST}.tmp" "${DST}"
echo "[sync_model_registry] wrote ${DST}"
