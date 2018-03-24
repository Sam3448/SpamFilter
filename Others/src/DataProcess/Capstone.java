package DataProcess;

import java.io.*;
import java.util.*;

public class Capstone {
	
	static String source = "/Users/SamZhang/Documents/Capstone/dataset/SMS.txt";
	static String hamTrain = "/Users/SamZhang/Documents/Capstone/dataset/sms/train/ham/sms_train.ham";
	static String hamTest = "/Users/SamZhang/Documents/Capstone/dataset/sms/test/ham/sms_test.ham";
	static String spamTrain = "/Users/SamZhang/Documents/Capstone/dataset/sms/train/spam/sms_train.spam";
	static String spamTest = "/Users/SamZhang/Documents/Capstone/dataset/sms/test/spam/sms_test.spam";
	static final int hamsection = 2, spamsection = 10;//每10个数据就有一个变成test rate: 9:1

	public static void main(String[] args) throws IOException{
		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(source))));
		FileWriter fwhamtrain = new FileWriter(new File(hamTrain)), fwspamtrain = new FileWriter(new File(spamTrain));
		FileWriter fwhamtest = new FileWriter(new File(hamTest)), fwspamtest = new FileWriter(new File(spamTest));
		
		int countHam = 0, countSpam = 0;
		int hamTest = 0, spamTest = 0;
		
		String line = "";
		while((line = br.readLine()) != null){
			String[] temp = line.split("\t");
			if(temp[0].equals("ham")){
				if(++countHam % hamsection == 0){
					fwhamtest.write(temp[1] + "\n");
					hamTest++;
				}
				else fwhamtrain.write(temp[1] + "\n");
			}
			else if(temp[0].equals("spam")){
				if(++countSpam % spamsection == 0){
					fwspamtest.write(temp[1] + "\n");
					spamTest++;
				}
				else fwspamtrain.write(temp[1] + "\n");
			}
		}
		
		br.close();
		fwspamtrain.close();
		fwspamtest.close();
		fwhamtrain.close();
		fwhamtest.close();
		
		System.out.println(String.format("Train ham : %d, test ham : %d", countHam - hamTest, hamTest));
		System.out.println(String.format("Train spam : %d, test spam : %d", countSpam - spamTest, spamTest));
	}

}
