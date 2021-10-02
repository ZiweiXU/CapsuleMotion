#!/bin/bash
# run this with watch -n60 -t --color ./listgpu.sh
thres=$1
if [[ -z "$thres" ]]; then thres=6 ; fi

calc() {
    echo - | awk "{print $1}"
}

SERVER_LIST=( cgpa0 cgpa1 cgpa2 cgpa3 )
RAMAIN_MEMORY_LIMIT=$(calc $thres*1024)
USER_NAME="ziwei-xu"
query_time=$(date)
echo "$query_time"
printf "SERVER\t GPU\t REMAIN_M/TOTAL_M\t UTIL\t POWER/MAXPWR \t\t USER\n"
for server in "${SERVER_LIST[@]}" ; do
    gpu_info=$(ssh -o StrictHostKeyChecking=no -o PasswordAuthentication=no "$server" gpustat --json)
    if [ "$?" -eq "0" ] ; then        
        gpu_num=$(echo "$gpu_info" | jq ".gpus | length")
        for gpu in $(seq 0 1 "$(calc $gpu_num-1)") ; do
            gpu_mem_total=$(echo "$gpu_info" | jq "$(printf '.gpus[%s]["memory.total"]' $gpu)")
            gpu_mem_used=$(echo "$gpu_info" | jq "$(printf '.gpus[%s]["memory.used"]' $gpu)")
            gpu_power_limit=$(echo "$gpu_info" | jq "$(printf '.gpus[%s]["enforced.power.limit"]' $gpu)")
            gpu_power_draw=$(echo "$gpu_info" | jq "$(printf '.gpus[%s]["power.draw"]' $gpu)")
            gpu_util=$(echo "$gpu_info" | jq "$(printf '.gpus[%s]["utilization.gpu"]' $gpu)")
            gpu_mem_remain=$(calc "$gpu_mem_total-$gpu_mem_used")
            proc_num=$(echo $gpu_info | jq "$(printf '.gpus[%s][\"processes\"] | length' $gpu)")
            proc_usr=""            
            proc_my=""
            for proc in $(seq 0 1 "$(calc $proc_num-1)"); do 
                this_usr=$(echo $gpu_info | jq "$(printf '.gpus[%s][\"processes\"][%s][\"username\"]' $gpu $proc)")
                if [[ $this_usr = *"$USER_NAME"* ]] ; then this_usr="\\e[32m$this_usr\\e[0m"; fi
                this_pid=$(echo $gpu_info | jq "$(printf '.gpus[%s][\"processes\"][%s][\"pid\"]' $gpu $proc)")
                proc_usr="$proc_usr"$this_usr,
            done

            if [[ $gpu_mem_remain -gt $RAMAIN_MEMORY_LIMIT ]]; then
                gpu="\\e[1;36m$gpu\\e[0m"
                this_server="\\e[1;36m$server\\e[0m"
            else
                gpu=$gpu
                this_server=$server
            fi
        printf "%b\t %b\t %5s/%-5s \t\t %-3s\t %4s/%-4s \t\t %b\n" "$this_server" "$gpu" "$gpu_mem_remain" "$gpu_mem_total" "$gpu_util" "$gpu_power_draw" "$gpu_power_limit" "$proc_usr"
        done
    else
        printf "%b\t \\e[1;35mConnection Error\\e[0m\n" "$server"
    fi
done