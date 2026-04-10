#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: bash case39_run_all.sh /path/to/case39_pipeline.env"
  exit 2
fi

ENV_FILE="$1"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "找不到配置文件: $ENV_FILE"
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${REPO_ROOT:?必须在 env 里设置 REPO_ROOT}"
: "${CASE_NAME:=case39}"
: "${PATCH_SCRIPT:=$REPO_ROOT/case39_first_batch_patch.py}"
: "${RUN_PATCH:=1}"
: "${STAGE_MODE:=symlink}"
: "${FORCE_STAGE:=0}"
: "${RUN_GENERIC_SMOKE:=1}"
: "${RUN_ORACLE_CONFIRM:=1}"
: "${WORKDIR:=.}"
: "${CLEAN_SRC:?必须设置 CLEAN_SRC}"
: "${ATTACK_SRC:?必须设置 ATTACK_SRC}"
: "${TRAIN_SRC:?必须设置 TRAIN_SRC}"
: "${VAL_SRC:?必须设置 VAL_SRC}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "缺少文件: $p"
    exit 3
  fi
}

require_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "缺少目录: $p"
    exit 3
  fi
}

run_cmd() {
  printf '>>> '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

require_dir "$REPO_ROOT"
require_file "$CLEAN_SRC"
require_file "$ATTACK_SRC"
require_file "$TRAIN_SRC"
require_file "$VAL_SRC"
if [[ -n "${TEST_SMOKE_SRC:-}" ]]; then
  require_file "$TEST_SMOKE_SRC"
fi

if [[ "$RUN_PATCH" == "1" ]]; then
  require_file "$PATCH_SCRIPT"
fi

if [[ "$RUN_ORACLE_CONFIRM" == "1" ]]; then
  : "${LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR:?RUN_ORACLE_CONFIRM=1 时必须设置 LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR}"
  : "${LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR:?RUN_ORACLE_CONFIRM=1 时必须设置 LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR}"
  : "${CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON:?RUN_ORACLE_CONFIRM=1 时必须设置 CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON}"
  require_dir "$LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR"
  require_dir "$LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR"
  require_file "$CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON"
fi

cd "$REPO_ROOT"
export DDET_CASE_NAME="$CASE_NAME"
STAMP="/tmp/${CASE_NAME}_bridge_$(date +%s).stamp"
touch "$STAMP"
log "STAMP=$STAMP"

if [[ "$RUN_PATCH" == "1" ]]; then
  log "应用第一批 patch"
  run_cmd python "$PATCH_SCRIPT" "$REPO_ROOT"
else
  log "跳过 patch 应用"
fi

FORCE_ARGS=()
if [[ "$FORCE_STAGE" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

log "staging canonical case39 资产"
CMD=(python prepare_case_bank_assets.py
  --case_name "$CASE_NAME"
  --out_root "metric/$CASE_NAME"
  --clean_src "$CLEAN_SRC"
  --attack_src "$ATTACK_SRC"
  --train_src "$TRAIN_SRC"
  --val_src "$VAL_SRC"
  --mode "$STAGE_MODE")
if [[ -n "${TEST_SMOKE_SRC:-}" ]]; then
  CMD+=(--test_src "$TEST_SMOKE_SRC")
fi
CMD+=("${FORCE_ARGS[@]}")
run_cmd "${CMD[@]}"

log "生成 v1/v2 manifest-only"
run_cmd python make_phase3_confirm_manifest.py \
  --workdir "$WORKDIR" \
  --case_name "$CASE_NAME" \
  --output_dir "metric/$CASE_NAME/phase3_confirm_blind_v1" \
  --manifest_only

run_cmd python make_phase3_confirm_manifest_v2.py \
  --workdir "$WORKDIR" \
  --case_name "$CASE_NAME" \
  --output_dir "metric/$CASE_NAME/phase3_confirm_blind_v2" \
  --manifest_only

if [[ -n "${LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR:-}" ]]; then
  log "staging blind confirm v1 holdout test banks"
  run_cmd python prepare_case_bank_assets.py \
    --case_name "$CASE_NAME" \
    --out_root "metric/$CASE_NAME" \
    --skip_canonical \
    --manifest "metric/$CASE_NAME/phase3_confirm_blind_v1/manifest.json" \
    --holdout_src_dir "$LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR" \
    --mode "$STAGE_MODE" \
    "${FORCE_ARGS[@]}"
fi

if [[ -n "${LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR:-}" ]]; then
  log "staging blind confirm v2 holdout test banks"
  run_cmd python prepare_case_bank_assets.py \
    --case_name "$CASE_NAME" \
    --out_root "metric/$CASE_NAME" \
    --skip_canonical \
    --manifest "metric/$CASE_NAME/phase3_confirm_blind_v2/manifest.json" \
    --holdout_src_dir "$LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR" \
    --mode "$STAGE_MODE" \
    "${FORCE_ARGS[@]}"
fi

if [[ "$RUN_GENERIC_SMOKE" == "1" ]]; then
  if [[ -e "metric/$CASE_NAME/mixed_bank_test_smoke.npy" ]]; then
    log "运行 generic holdout wiring smoke"
    run_cmd python evaluation_budget_scheduler_phase3_holdout.py \
      --clean_bank "metric/$CASE_NAME/metric_clean_alarm_scores_full.npy" \
      --attack_bank "metric/$CASE_NAME/metric_attack_alarm_scores_400.npy" \
      --train_bank "metric/$CASE_NAME/mixed_bank_fit.npy" \
      --val_bank "metric/$CASE_NAME/mixed_bank_eval.npy" \
      --test_bank "metric/$CASE_NAME/mixed_bank_test_smoke.npy" \
      --slot_budget_list 1 2 \
      --decision_step_group 1 \
      --busy_time_quantile 0.65 \
      --max_wait_steps 10 \
      --output "metric/$CASE_NAME/budget_scheduler_phase3_holdout_smoke.npy"
  else
    log "跳过 generic smoke：未提供 TEST_SMOKE_SRC"
  fi
fi

if [[ -n "${LOCAL_CASE39_HOLDOUT_BANKS_V1_DIR:-}" ]]; then
  log "生成 v1 baseline holdout summaries"
  run_cmd python make_phase3_confirm_manifest.py \
    --workdir "$WORKDIR" \
    --case_name "$CASE_NAME" \
    --output_dir "metric/$CASE_NAME/phase3_confirm_blind_v1"
fi

if [[ -n "${LOCAL_CASE39_HOLDOUT_BANKS_V2_DIR:-}" ]]; then
  log "生成 v2 baseline holdout summaries"
  run_cmd python make_phase3_confirm_manifest_v2.py \
    --workdir "$WORKDIR" \
    --case_name "$CASE_NAME" \
    --output_dir "metric/$CASE_NAME/phase3_confirm_blind_v2"
fi

if [[ "$RUN_ORACLE_CONFIRM" == "1" ]]; then
  log "运行 oracle_confirm v1"
  run_cmd python run_phase3_oracle_confirm.py \
    --manifest "metric/$CASE_NAME/phase3_confirm_blind_v1/manifest.json" \
    --dev_screen_summary "$CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON" \
    --output "metric/$CASE_NAME/phase3_oracle_confirm_v1"

  log "运行 oracle_confirm v2"
  run_cmd python run_phase3_oracle_confirm.py \
    --manifest "metric/$CASE_NAME/phase3_confirm_blind_v2/manifest.json" \
    --dev_screen_summary "$CASE14_FROZEN_ORACLE_WINNER_SUMMARY_JSON" \
    --output "metric/$CASE_NAME/phase3_oracle_confirm_v2"
fi

log "防回归检查：查 metric/case14 是否有新写入"
find metric/case14 -type f -newer "$STAMP" -print || true

log "防回归检查：查 case39 输出是否残留 metric/case14 路径"
grep -R "metric/case14" \
  "metric/$CASE_NAME/phase3_confirm_blind_v1" \
  "metric/$CASE_NAME/phase3_confirm_blind_v2" \
  "metric/$CASE_NAME"/*.json -n || true

log "关键输出建议检查"
for p in \
  "metric/$CASE_NAME/asset_protocol.json" \
  "metric/$CASE_NAME/bridge_contract.json" \
  "metric/$CASE_NAME/phase3_confirm_blind_v1/manifest.json" \
  "metric/$CASE_NAME/phase3_confirm_blind_v2/manifest.json" \
  "metric/$CASE_NAME/phase3_oracle_confirm_v1/aggregate_summary.json" \
  "metric/$CASE_NAME/phase3_oracle_confirm_v2/aggregate_summary.json"; do
  if [[ -e "$p" ]]; then
    echo "OK: $p"
  else
    echo "MISSING: $p"
  fi
done

log "全部步骤执行完毕"
