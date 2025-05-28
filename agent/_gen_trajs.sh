#!/bin/bash
sweagent run-batch --num_workers 20 \
    --instances.deployment.docker_args=--memory=10g \
    --config agent/swesmith_gen_gpt2.yaml \
    --instances.path ~/SWEQ/logs/task_insts/ig_v2/test_openai1__ig_v2_n1.json \
    --output_dir trajectories/ehejin/new_RUN5 \
    --random_delay_multiplier=1 \
    --agent.model.temperature 0.0  
# Remember to set CLAUDE_API_KEY_ROTATION=key1:::key2:::key3
