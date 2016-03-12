rm -rf ~/data

mv data.tar.gz ~/
cd ~/

tar -zxvf data.tar.gz
tar -zxvf answers.tar.gz

chmod -R 777 ~/data
chmod -R 777 ~/answers

rm -rf ~/data.tar.gz
