#include <bits/stdc++.h>
using namespace std;



float threshold = 0.3;


int editDistance(string a, string b, int m, int n, vector<vector<int>>&dp){
    
    if(m==0) return n;
    if(n==0) return m;


    
    if(dp[m][n]!=-1) return dp[m][n];
    
    if(a[m-1]==b[n-1]){


        return dp[m][n] = editDistance(a,b,m-1,n-1,dp);
        
    } else {
        
        int x,y,z;
        
        x = (dp[m-1][n]==-1) ? editDistance(a,b,m-1,n,dp) : dp[m-1][n];
        
        y = (dp[m][n-1]==-1) ? editDistance(a,b,m,n-1,dp) : dp[m][n-1];
        
        z = (dp[m-1][n-1]==-1) ? editDistance(a,b,m-1,n-1,dp) : dp[m-1][n-1];


        
        return  dp[m][n] = 1 + min(x,min(y,z));

        

        
    }

    return dp[m][n];

    
}


vector<string> getWords(){

    vector<string>x;


    ifstream file("result.txt");
    string s;

    while(getline(file,s)){

        string temp=s.substr(0,10);

        if(temp=="Recognized"){

            temp=s.substr(13);

            temp.erase(temp.end()-1);

            x.push_back(temp);
            
        }

    }

    return x;

}



int main(){

    


    vector<string>userAns;



    userAns = getWords();

    for(int i=0;i<userAns.size();i++){

        for(int j=0;j<userAns[i].size();j++){
            
            if(isalpha(userAns[i][j]))userAns[i][j] = tolower(userAns[i][j]);
            

        }


    }

    vector<string> correct_ans_set1 = {"goa","mercury","tokyo","hyderabad","artic"};

    int score = 0;

    for(int i=0;i<5;i++){

        vector<vector<int>> dp (userAns[i].size()+1,vector<int>(correct_ans_set1[i].size()+1,-1));
        
        int e = editDistance(userAns[i],correct_ans_set1[i],userAns[i].size(),correct_ans_set1[i].size(),dp);

        int l = correct_ans_set1[i].size();



        
        int val = ceil((float)l * threshold);

        cout <<e<<" "<<val<<" "<<userAns[i]<<" "<<correct_ans_set1[i]<<endl;

        
        if(e<=val) score++;
        
    }

    cout <<"Score : "<<score<<"/"<<5<<endl;




}