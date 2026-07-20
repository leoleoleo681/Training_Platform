docker run --rm \
    --name train-task_test-first_test_docker \
    --gpus '"device=0,1"' \
    --shm-size=8g \
    --user "$(id -u lu.boyu):$(id -g lu.boyu)" \
    --mount type=bind,src=/mnt/sdb/user_home/lu.boyu/Training_Platform/docker_Training_Platform/Training_Platform/task_test,dst=/mnt/task \
    --cpus=8 \
    --memory=16g \
    --memory-swap=16g \
    --pids-limit=1024 \
    --security-opt=no-new-privileges \
    --read-only \
    --tmpfs /tmp \
    training-text-single:1.0.0 \
    train --task-type text_classification_single \
            --config /mnt/task/models/first_test_docker/run_train.json