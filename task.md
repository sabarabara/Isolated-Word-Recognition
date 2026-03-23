やりたいこと
競技かるたの音声切り取ってそこから100種類のどれの確率かを当てたい


まず0.5sで区切った
音声を傘増しして一つにつきノイズを10種類のせてデータ量 3*10
melして行列に変えてcnn1d cnn2d ast gru lstm conformerをやった

0.5sだからrnn astは弱い
基本的にcnnでやるのが一番強い

精度がcnnだとtop-1で55%だったからpoolingで100にクラス化する前にtransformerに入れたらなんかうまくいく気がした
conformerの精度は低い


やれそうなこと


melのCNN2Dがなんか一番精度高い なんかくっつけたらもっと精度上がらないかな conformerをいじる
dropout_conv = float(cfg.get("conformer_dropout_conv", 0.2)) -> 0.05にしてみる
0.5sからどんどん小さくなってくからそもそもFFTが向いてないかも そのままの特徴量でぶち込むのはどうなのか os STFT
時間領域を引き延ばしてうまくいかなかったRNN ASTでやる
これそもそも数理モデルじゃダメなのか100択だから当たりそう