python_exec="venv/bin/python"
sweepfile="$1"
noconfirm=1
resuming=0

echo $@
if [ "${sweepfile}" = "resume" ]; then
  [[ $# -lt 2 ]] && echo "Need sweep id to resume" && exit 1
  sweepid="$2"
  cmd="$python_exec -m wandb sweep --resume ${sweepid}"
  resuming=1

else
  [[ ! -f ${sweepfile} ]] && echo "Can't find sweep file at [${sweepfile}]" && exit 1
  cmd="$python_exec -m wandb sweep ${sweepfile}"
fi

echo "Running:[$cmd]"
res=$($cmd 2>&1)
echo "Output:"
echo "$res"
if [ ! "${sweepfile}" = "resume" ]; then
  agent_id=$(echo "$res" | grep 'Run sweep agent with:' | awk '{print $NF}')
  agent_cmd="$python_exec -m wandb agent $agent_id"
  echo "Run agent command: [${agent_cmd}]"
  echo "Any key to continue or C-c to abort"
  [[ -z $noconfirm ]] && read
  $agent_cmd
fi

