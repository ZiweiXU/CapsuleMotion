#!/usr/bin/env bash

waitpid () {
    PID="$1"
    if [[ "$PID" == 0 ]]; then return ; fi
    while s=$(ps -p "$PID" -o s=) && [[ "$s" && "$s" != 'Z' ]]; do
        sleep 1
    done
}

calc () {
    echo - | awk "{print $1}"
}

elementIn () {
    local e match="$1"
    shift
    for e; do [[ "$e" == "$match" ]] && return 0; done
    return 1
}

join_by () { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

elementMax () {
    local array=$1
    local k=$2
    if [[ -z $k ]]; then k=1 ; fi
    result=$(python3 -c "\
a = [$(join_by , "${array[@]}")];\
a = [(i,a[i]) for i in range(len(a))];\
b = sorted(a, key=lambda x:x[1])[-${k}:];\
ids = [i[0] for i in b]
vals = [i[1] for i in b]
ids = ','.join(map(str, ids));\
vals = ','.join(map(str, vals));\
print('{} {}'.format(ids, vals), end='');")
    echo "$result"
}

elementMin () {
    local array=$1
    local k=$2
    if [[ -z $k ]]; then k=1 ; fi
    result=$(python3 -c "\
a = [$(join_by , "${array[@]}")];\
a = [(i,a[i]) for i in range(len(a))];\
b = sorted(a, key=lambda x:x[1])[:${k}];\
ids = [i[0] for i in b]
vals = [i[1] for i in b]
ids = ','.join(map(str, ids));\
vals = ','.join(map(str, vals));\
print('{} {}'.format(ids, vals), end='');")
    echo "$result"
}

wait_on_lock () {
    local lock_name="$1"
    while true; do
        if [[ -f "$lock_name" ]]; then
            sleep "$(calc "$RANDOM / 32767 * 5")"
        else
            break
        fi
    done
}


get_gpu () {
    local mem_req="$1"
    local req_cards="$2"
    local lock_file="$HOME/.gpu.lock"
    if [[ -z $req_cards ]]; then req_cards=1; fi
    sleep "$(calc "$RANDOM / 32767 * 5")"
    # wait for lock file to be removed
    wait_on_lock "$lock_file"
    # create lock so that other get_gpu waits
    # the lock file should be removed in python script by which time GPU utility
    # should be stable so that waiting jobs can proceed to get_gpu
    touch "$lock_file"
    # query and return the right gpu
    while true; do
        sleep "$(calc "$RANDOM / 32767 * 5")"
        mapfile -t mem_avail < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        mapfile -d ' ' -t best_info < <( elementMax "$(join_by , "${mem_avail[@]}")" ${req_cards} )
        best_ids=${best_info[0]}
        best_mem=${best_info[1]}
        
        mapfile -d ' ' -t min_best < <( elementMin "${best_mem}" )
        if [[ ${min_best[1]} -gt $mem_req ]]; then
            echo "${best_ids}"
            break
        fi
    done
}
