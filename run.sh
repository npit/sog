[[ -z $PYTHON_EXEC ]] && PYTHON_EXEC="python"
$PYTHON_EXEC  run.py "--tracker" "wandb" "--num_episodes" "10" "--no_eat_penalty" "-1" "--num_episodes" "5" "--hunger_delta" "1" "--preset" "random3" "--run_name" "test_run"
