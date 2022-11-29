# Check the type of transformer
cur_file='/workspace/bert/modeling.py'
ori_file='/workspace/bert/modeling_ori.py'
tempo_file='/workspace/bert/modeling_tempo.py'

if diff -q $cur_file $ori_file; then 
    echo " * ==> Original Transformer" 
fi

if diff -q $cur_file $tempo_file; then
    echo "* ==> Tempo Transformer" 
fi