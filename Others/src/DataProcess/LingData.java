package DataProcess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class LingData {
	
	static String sourceFolder = "/Users/SamZhang/Documents/Capstone/dataset/lingspam_public/lemm_stop/";
	static String hamTrain = "/Users/SamZhang/Documents/Capstone/dataset/ling/train/ham/ling_train.ham";
	static String hamTest = "/Users/SamZhang/Documents/Capstone/dataset/ling/test/ham/ling_test.ham";
	static String spamTrain = "/Users/SamZhang/Documents/Capstone/dataset/ling/train/spam/ling_train.spam";
	static String spamTest = "/Users/SamZhang/Documents/Capstone/dataset/ling/test/spam/ling_test.spam";
	static final int hamsection = 2, spamsection = 10;//每10个数据就有一个变成test rate: 9:1

	public static void main(String[] args) throws IOException{
		List<File> files = getListFiles();
		System.out.println(files.size());
		
		FileWriter fwhamtrain = new FileWriter(new File(hamTrain)), fwspamtrain = new FileWriter(new File(spamTrain));
		FileWriter fwhamtest = new FileWriter(new File(hamTest)), fwspamtest = new FileWriter(new File(spamTest));
		
		int countHam = 0, countSpam = 0;
		int hamTest = 0, spamTest = 0;
		
		for(File f : files){
			String curFileName = f.getName();
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
			String line = "";
			StringBuilder sb = new StringBuilder();
			while((line = br.readLine()) != null){
				line = line.trim();
				if(line.isEmpty()) continue;
				if(line.startsWith("Subject")){
					sb.append(line.substring(line.indexOf(":") + 1)).append(" ");
				}
				else sb.append(line);
			}
			
			if(!curFileName.startsWith("spm")){
				if(++countHam % hamsection == 0){
					fwhamtest.write(sb.toString() + "\n");
					hamTest++;
				}
				else fwhamtrain.write(sb.toString() + "\n");
			}
			else if(curFileName.startsWith("spm")){
				if(++countSpam % spamsection == 0){
					fwspamtest.write(sb.toString() + "\n");
					spamTest++;
				}
				else fwspamtrain.write(sb.toString() + "\n");
			}
		}
		
		fwspamtrain.close();
		fwspamtest.close();
		fwhamtrain.close();
		fwhamtest.close();
		
		System.out.println(String.format("Train ham : %d, test ham : %d", countHam - hamTest, hamTest));
		System.out.println(String.format("Train spam : %d, test spam : %d", countSpam - spamTest, spamTest));
		
	}
	public static List<File> getListFiles() throws IOException{
		List<File> res = new ArrayList();
		File root = new File(sourceFolder);
		File[] subFolder = root.listFiles();
		for(File folder : subFolder){
			if(folder.isDirectory()){
				File[] files = folder.listFiles();
				for(File f : files){
					if(!f.isFile()) continue;
					res.add(f);
				}
			}
		}
		return res;
	}
}
