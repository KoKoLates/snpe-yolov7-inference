#! /bin/bash

print_color_message() {
    color=$1
    message=$2
    echo -e "\e[${color}m${message}\e[0m"
}

progress() {
    progress=$1
    length=25
    block=$(((progress * length) / 100))

    # Print the progress bar
    printf "["
    printf "%${block}s" | tr ' ' '#'
    printf "%$((length - block))s" | tr ' ' '-'
    printf "] %d%%\r" "$progress"
}

# restart the voxl-streamer services
print_color_message "35" "[info] voxl-streamer: starting the server services"
systemctl restart voxl-streamer

sleep 1
# Run the snpe_test command with progress
for ((i = 0; i <= 100; i += 10)); do
    progress_bar $i
    sleep 0.1
done
echo ""

# start the snpe inference task
print_color_message "32" "[info] running snpe inference..."
./snpe_task
