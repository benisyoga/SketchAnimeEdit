卒業研究の際に自作したモデルのファイルです（スクリプトファイルのみ）。  

各ファイルの大まかな説明  
make_dataset.py...立ち絵画像を学習用に前処理する。  
network.py...モデルのネットワークをまとめたファイル。  
network_module.py...モデルのネットワークに使用するモジュールをまとめたファイル。  
run_train.sh...学習を実行するシェルスクリプト。  
run_test.sh...推論を実行するシェルスクリプト。  
test.py...実際に推論を行う。  
test_dataset.py...推論用のデータセットを読み込む。  
tester.py...推論時のパラメータを設定する。  
train.py...実際に学習を行う。  
train_dataset.py...学習用のデータセットを読み込む。  
trainer.py...学習時のパラメータを設定する。  
utils.py...その他機能（複雑な損失関数の計算など）を行う。  
