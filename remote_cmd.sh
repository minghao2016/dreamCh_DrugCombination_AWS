rm -rf /mina
mkdir /mina

mv data.tar.gz /mina
mv answers.tar.gz /mina
cd /mina
tar -zxvf data.tar.gz
tar -zxvf answers.tar.gz

chmod -R 777 /mina
