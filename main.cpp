//
//  main.cpp
//  C++ implementation of Eriksen model
//
//  Original implementation by Nick Yeung
//  Changes by Sander Nieuwenhuis & Roy de Kleijn
//
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// Global constants - general stuff
const int NUM_UNITS = 11;
const int CORRECT = 1;  const int ERROR = 0;
const int NONE = -1; const int TRUE = 1; const int FALSE = 0;
const int LEFT = 6; const int RIGHT = 7;
const int RUN_CYCLES = 50;
const int CM = 0; const int IC = 1;
const int H = 0; const int S = 1;
const int CONT_PROC = 6;
const int WARM_UPS = 10; // Number of trials to discard (while levels of control input stabilise)

// Unit activation dynamics
const float MIN_ACT = -0.2;
const float MAX_ACT = 1.0;
const float REST_ACT = -0.1;
const float DECAY = 0.1;

// Network parameters
const float ALPHA = 0.08;
const float GAMMA = 0.12;
const float ESR_H = 1.5 * ALPHA; // Strength of excitation from stimulus to response - default = 1.5
const float ESR_S = 1.5 * ALPHA; // Strength of excitation from stimulus to response - default = 1.5
const float ESA = 2.0 * ALPHA; // Strength of excitation between stimulus and attention - default = 2.0

// Variable network parameters
float IN_BIAS = 0.0; 		// Bias on stimulus units - default = 0.0
float RES_H_BIAS = 0.0; 	// Bias on H response unit - default = 0.0
float STIM_H_BIAS = 0.0; 	// Bias on H stimulus unit - default = 0.0
float TASK_BIAS = 0.0; 	// Bias on attention units
float LiR = -3.0 * GAMMA; 	// Strength of lateral inhibition in response layer - default = -3.0
float LiS = -2.0 * GAMMA; 	// Strength of lateral inhibition in stimulus layer - default = -2.0
float LiA = -1.0 * GAMMA; 	// Strength of lateral inhibition in attention layer - default = -1.0

// Strength of external inputs
float estr = 0.4;
float ext_sH = 0.15 + STIM_H_BIAS; 	// Signal - i.e. stimulus presented
float ext_sS = .15; 			// Signal - i.e. stimulus presented
float ext_nH = 0.0; float ext_nS = 0.0;	// Noise - i.e. stimulus not presented
float ext_C = 0.2;
float ext_F = 0;
float ext_rH = .03; float ext_rS = .03;

// Unit gains - default value of 1.0 for all
float gIH = 1.0; float gIHc = gIH; float gIS = gIH; float gISc = gIH; float gRH = 1.0; float gRS = 1.0; float gTf = 1.0; float gTc = 1.0;

float threshold_H = .18; 		// Response threshold for H response (left) - default = 0.18
float threshold_S = threshold_H; 	// Response threshold for S response (right) - default = 0.18

// Global variables
float ext_input[NUM_UNITS];
float new_input[NUM_UNITS];
float net_input[NUM_UNITS];
float activation[NUM_UNITS];
float unit_gain[NUM_UNITS]={ gIH,gIHc,gIH, gIS,gISc,gIS, gRH,gRS, gTf,gTc,gTf };

// Simulation parameters, set in main() according to model author
// These defaults are Matt Botvinick's:
char author[4];
float noise_val;
int prime_cycles;
int units_to_prime = 2; 		// This number specifies which row with units to prime in an array
int onset_delay; 			// Spencer & Coles delayed stimulus presentation by 13 cycles into the RT
int attn_delay;             // Delay external input to center attention unit (RdK)
int cont_proc;
int attentionType = 2;     // Set attention activation to fixed values? (0 = default, 1 = fixed values, 2 = low during warning period)

float LRP_signal[RUN_CYCLES], corrRAct[RUN_CYCLES], incorrRAct[RUN_CYCLES];

float leftATTNact[RUN_CYCLES], ctrATTNact[RUN_CYCLES], rightATTNact[RUN_CYCLES];  // RdK: want to write to file

// The units are: wR wG wX cR cG cX rR rG tW tC
float weights[NUM_UNITS][NUM_UNITS]={ 			// Weights from stimulus units
    0,LiS,LiS, LiS,LiS,LiS, ESR_H,0, ESA,0,0,			// H left
    LiS,0,LiS, LiS,LiS,LiS, ESR_H,0, 0,ESA,0, 		// H centre
    LiS,LiS,0, LiS,LiS,LiS, ESR_H,0, 0,0,ESA, 		// H right
    LiS,LiS,LiS, 0,LiS,LiS, 0,ESR_S, ESA,0,0,			// S left
    LiS,LiS,LiS, LiS,0,LiS, 0,ESR_S, 0,ESA,0, 			// S centre
    LiS,LiS,LiS, LiS,LiS,0, 0,ESR_S, 0,0,ESA,			// S right
    0,0,0, 0,0,0, 0,LiR, 0,0,0, 				// Weight from H response unit
    0,0,0, 0,0,0, LiR,0, 0,0,0, 				// Weight from S response unit
    ESA,0,0, ESA,0,0, 0,0, 0,LiA,LiA,			// Weight from left attention unit
    0,ESA,0, 0,ESA,0, 0,0, LiA,0,LiA, 			// Weight from centre attention unit
    0,0,ESA, 0,0,ESA, 0,0, LiA,LiA,0			// Weight from right attention unit
};

float prime_units[NUM_UNITS];
float trial_types[2][2][NUM_UNITS];

// Functions
void run_sim(int sims, int num_trials);
void run_trial(int target, int condition, float trial_results[3]);
void run_cycles(float the_inputs[NUM_UNITS+2], int max_cycles, int primeOrReal, float trial_results[3]);
float getNoise();
int irand(int n);

int badtrials; // Bad trial counter - RdK

/************************************ MAIN FUNCTION *************************************/
int main()
{
    char decision[4]; int seed, sim, sims, num_trials;
    do {
        printf("Botvinick or Spencer parameters? (b/s): "); scanf("%s", author);
        
        // Set the parameters accordingly
        if(author[0]=='s') 			// Spencer parameters
        {
            noise_val=0.036; 		// default = 0.036
            prime_cycles=100; 		// default = 100
            units_to_prime=0; 		// default = 0
            printf("What delay in stimulus presentation?: "); scanf("%d", &onset_delay);
            
            printf("What delay in attention priming?: "); scanf("%d", &attn_delay);
            
            estr=0.28;          // default = 0.28
            ext_C=0.23; 		// default = 0.23
            ext_F=0.0;
            ext_sH = ext_sS = 0.14;	 // default = 0.14
            ext_nH = ext_nS = 0.0;	// default = 0.0
        }
        else 				// Botvinick parameters
        {
            noise_val=0.035;
            prime_cycles=3; 		// default = 3
            units_to_prime=1; 		// default = 1
            onset_delay=0;
            estr=0.4;
            ext_C=0.23;
            ext_F=0;
        }
        //seed=irand(1000); sims=1; num_trials=20000;
        printf("Enter random seed: "); scanf("%d", &seed);
        srand(seed);
        do
        {
            printf("How many simulations (max 100): "); scanf("%d", &sims);
        } while(sims>100);
        
        printf("How many trials per simulation: "); scanf("%d", &num_trials);
        
        // Added to make threshold adjustable (RdK)
        printf("Enter new threshold value if necessary (default = .18, N&dK = .25): "); scanf("%f", &threshold_H);
        threshold_S = threshold_H;
        
        for(sim=0; sim<sims; sim++)
        {
            printf("Running %d trials...\n\n", num_trials);
            float init_unit_gain[NUM_UNITS]={ gIH,gIHc,gIH, gIS,gISc,gIS, gRH,gRS, gTf,gTc,gTf };
            
            for(int i=0; i<NUM_UNITS; i++)
                unit_gain[i]=init_unit_gain[i];
            
            float init_trial_types[2][2][NUM_UNITS]=
            {
                ext_sH,ext_sH,ext_sH, ext_nS,ext_nS,ext_nS, ext_rH,ext_rS, ext_F,ext_C,ext_F,  // H compatible
                ext_nH,ext_sH,ext_nH, ext_sS,ext_nS,ext_sS, ext_rH,ext_rS, ext_F,ext_C,ext_F,  // H incompatible
                ext_nH,ext_nH,ext_nH, ext_sS,ext_sS,ext_sS, ext_rH,ext_rS, ext_F,ext_C,ext_F,  // S compatible
                ext_sH,ext_nH,ext_sH, ext_nS,ext_sS,ext_nS, ext_rH,ext_rS, ext_F,ext_C,ext_F,  // S incompatible
            };
            
            for(int tt1=0; tt1<2; tt1++)
                for(int tt2=0; tt2<2; tt2++)
                    for(int tt3=0; tt3<NUM_UNITS; tt3++)
                        trial_types[tt1][tt2][tt3]=init_trial_types[tt1][tt2][tt3];
            
            run_sim(sim, num_trials);
        } // end of sim loop
        
        printf("Again? (y/n): "); scanf("%s", decision);
        
    } while (decision[0]!='n');	// end of do loop
    return 0;
}


/************************************ PROGRESS BAR FUNCTION *************************************/

// Process has done i out of n rounds,
// and we want a bar of width w and resolution r.
static inline void loadBar(int x, int n, int r, int w)
{
    
    
    // Only update r times.
    if ( x % (n/r) != 0 ) return;
    
    // Calculate the ratio of complete-to-incomplete.
    float ratio = x/(float)n;
    int   c     = ratio * w;
    
    // Show the percentage complete.
    printf("%3d%% [", (int)(ratio*100) );
    
    // Show the load bar.
    for (int x=0; x<c; x++)
        printf("=");
    
    for (int x=c; x<w; x++)
        printf(" ");
    
    // ANSI Control codes to go back to the
    // previous line and clear it.
    printf("]\r"); // Move to the first column
    fflush(stdout);
}


/************************************ RUN_SIMS FUNCTION *************************************/

void run_sim(int sim, int num_trials)
{
	int target, condition, trial, accuracy;
	float trial_results[3], sum_RT[2][2], error_RT, correct_RT, err_count, corr_count;
	int RT_count[2][2], trial_count[2][2], min_RT[2][2], max_RT[2][2], null[2][2];
	float error_count[2][2], stimulus_error_count[2][2];
	float gratton_RT[2][2], gratton_ER[2][2]; int gratton_count[2][2], gratton_RT_count[2][2], prev_cond;
    
	/* Open the data files */
	FILE *LRP_OUT; char LRP_filename[10]; sprintf(LRP_filename, " LRP.%d", sim);
	if ((LRP_OUT=fopen(LRP_filename, "w"))==NULL) { printf("Cannot open Raw_data file.\n"); exit(1); }
    
    FILE *ATTN1_OUT; char ATTN1_filename[10]; sprintf(ATTN1_filename, "ATTN1.%d", sim);
    if ((ATTN1_OUT=fopen(ATTN1_filename, "w"))==NULL) { printf("Cannot open Raw_data file.\n"); exit(1); }
    
    FILE *ATTN2_OUT; char ATTN2_filename[10]; sprintf(ATTN2_filename, "ATTN2.%d", sim);
    if ((ATTN2_OUT=fopen(ATTN2_filename, "w"))==NULL) { printf("Cannot open Raw_data file.\n"); exit(1); }
    
    FILE *ATTN3_OUT; char ATTN3_filename[10]; sprintf(ATTN3_filename, "ATTN3.%d", sim);
    if ((ATTN3_OUT=fopen(ATTN3_filename, "w"))==NULL) { printf("Cannot open Raw_data file.\n"); exit(1); }
    
    
	corr_count=err_count=error_RT=correct_RT=0;
    
	for(target=0; target<2; target++)
        for(condition=0; condition<2; condition++)
        {
            sum_RT[target][condition]=RT_count[target][condition]=error_count[target][condition]=0;
            trial_count[target][condition]=null[target][condition]=0;
            min_RT[target][condition]=1000; max_RT[target][condition]=0;
            gratton_RT[target][condition]=gratton_ER[target][condition]=gratton_count[target][condition]=0;
            gratton_RT_count[target][condition]=stimulus_error_count[target][condition]=0;
        }
    
	prev_cond=0;
	for(trial=0; trial<num_trials+WARM_UPS; trial++)
	{
        loadBar(trial,num_trials,5,10); // Show progress bar
        
		// Decide the target and condition for the trial
		if (irand(4)>1) target = H; else target = S;
		if (irand(4)>1) condition = CM; else condition = IC;
        
		trial_types[target][condition][8]=trial_types[target][condition][10]=ext_F;
		trial_types[target][condition][9]=ext_C;
        
		// Run it ...
		run_trial(target, condition, trial_results);
		accuracy=trial_results[1];
        
		// Record the results (excluding warm-up trials).
		if(trial>(WARM_UPS-1))
		{
			trial_count[target][condition]++;
			gratton_count[condition][prev_cond]++;
			if(accuracy==ERROR)
				stimulus_error_count[target][condition]++;
            
			if(trial_results[1]==CORRECT)
			{
				if(trial_results[0]<min_RT[target][condition])
					min_RT[target][condition]=trial_results[0];
				if(trial_results[0]>max_RT[target][condition])
					max_RT[target][condition]=trial_results[0];
				
				corr_count++;
				correct_RT+=(float) trial_results[0];
				RT_count[target][condition]++;
				sum_RT[target][condition]+=(float) trial_results[0];
				gratton_RT[condition][prev_cond]+=(float) trial_results[0];
				gratton_RT_count[condition][prev_cond]++;
			}
			else if (trial_results[1]==NONE)
			{
				null[target][condition]++;
				// printf("\n\n\n\n\nBAD TRIAL!!!!!!!!\n\n\n\n\n");
                badtrials++; // add to counter instead of printing to screen - RdK
			}
			else
			{
				err_count++;
				error_RT+=(float) trial_results[0];
				error_count[target][condition]++;
				gratton_ER[condition][prev_cond]++;
			}
		}
        
		// Write response unit activations and conflict signal to file
		if((num_trials%1000)!=1 && trial>(WARM_UPS-1))
		{
			int cycle;
            
			fprintf(LRP_OUT, "%d\t%d\t%d\t%d\t%5.3d\t", condition, target, (int) trial_results[0],
                    (int) trial_results[1], (int) trial_results[2]);
			for(cycle = 0; cycle < RUN_CYCLES; cycle++)
				fprintf(LRP_OUT, "%5.3f\t", LRP_signal[cycle]);
            
            fprintf(LRP_OUT, "\n");
            
            
            fprintf(ATTN1_OUT, "%d\t%d\t%d\t%d\t%5.3d\t", condition, target, (int) trial_results[0],
                    (int) trial_results[1], (int) trial_results[2]);
			for(cycle = 0; cycle < RUN_CYCLES; cycle++)
				fprintf(ATTN1_OUT, "%5.3f\t", leftATTNact[cycle]);
            
            fprintf(ATTN1_OUT, "\n");
            
            fprintf(ATTN2_OUT, "%d\t%d\t%d\t%d\t%5.3d\t", condition, target, (int) trial_results[0],
                    (int) trial_results[1], (int) trial_results[2]);
			for(cycle = 0; cycle < RUN_CYCLES; cycle++)
				fprintf(ATTN2_OUT, "%5.3f\t", ctrATTNact[cycle]);
            
            fprintf(ATTN2_OUT, "\n");
            
            fprintf(ATTN3_OUT, "%d\t%d\t%d\t%d\t%5.3d\t", condition, target, (int) trial_results[0],
                    (int) trial_results[1], (int) trial_results[2]);
			for(cycle = 0; cycle < RUN_CYCLES; cycle++)
				fprintf(ATTN3_OUT, "%5.3f\t", rightATTNact[cycle]);
            
            fprintf(ATTN3_OUT, "\n");
            
			
		}
        
		prev_cond=condition;
        
	} // end of trial loop
    
	char stimuli[2][2][4]= { "HHH", "SHS", "SSS", "HSH" };
	char gratton_cond[2][2][3] = { "cC", "iC", "cI", "iI" };
	float gratton_effect_RT, gratton_effect_ER;
    
	/* SCREEN OUTPUT */
	// RTs and error rates for the four different stimuli
	printf("\n");
	printf("ext_T =%5.2f; gTc =%5.2f; thresh = %5.3f\n", ext_sH, gTc, threshold_H);
	printf("Corr_RT = %5.2f Err_RT = %5.2f Bad trials: %d\n\n", (float) correct_RT/corr_count, (float) error_RT/err_count, badtrials);
    
	for(target=0; target<2; target++)
	{
		for(condition=0; condition<2; condition++)
		{
			printf("%s ... RT = %5.2f (%d-%d) ER = %5.1f \n", stimuli[target][condition],
                   (sum_RT[target][condition]/RT_count[target][condition]), min_RT[target][condition], max_RT[target][condition],
                   (float) (100*(error_count[target][condition]/trial_count[target][condition])));
		}
        
	}
	
	// Gratton effects in RTs, error rates and conflict signals
	for(condition=0; condition<2; condition++)
	{
		for(prev_cond=0; prev_cond<2; prev_cond++)
		{
			printf("%s = %5.2f ", gratton_cond[condition][prev_cond], (gratton_RT[condition][prev_cond]/gratton_RT_count[condition][prev_cond]));
		}
	}
	
	printf("\n");
	
	for(condition=0; condition<2; condition++)
	{
		for(prev_cond=0; prev_cond<2; prev_cond++)
		{
			printf(" %5.1f ", (float) (100*(gratton_ER[condition][prev_cond]/gratton_count[condition][prev_cond])));
		}
	}
	
	printf("\n");
    
	gratton_effect_RT=((gratton_RT[0][1]/gratton_RT_count[0][1])-(gratton_RT[0][0]/gratton_RT_count[0][0]));
	gratton_effect_RT+=((gratton_RT[1][0]/gratton_RT_count[1][0])-(gratton_RT[1][1]/gratton_RT_count[1][1]));
	gratton_effect_ER=((gratton_ER[0][1]/gratton_count[0][1])-(gratton_ER[0][0]/gratton_count[0][0]));
	gratton_effect_ER+=((gratton_ER[1][0]/gratton_count[1][0])-(gratton_ER[1][1]/gratton_count[1][1]));
	printf("Gratton_RT = %5.2f Gratton_ER = %5.2f\n\n", gratton_effect_RT, (gratton_effect_ER*100));
    
    /* Output mean RT for congruent and incongruent trials (RdK) */
    
    printf("\n");
    printf("Mean congruent RT:   %5.0f ms (%.0f ms)\n", (((sum_RT[0][0]/RT_count[0][0])+(sum_RT[1][0]/RT_count[1][0]))/2)*10, ((((sum_RT[0][0]/RT_count[0][0])+(sum_RT[1][0]/RT_count[1][0]))/2)*10) + 75);
    printf("Mean incongruent RT: %5.0f ms (%.0f ms)\n", (((sum_RT[0][1]/RT_count[0][1])+(sum_RT[1][1]/RT_count[1][1]))/2)*10, ((((sum_RT[0][1]/RT_count[0][1])+(sum_RT[1][1]/RT_count[1][1]))/2)*10 + 75));
    printf("Congruence effect:   %5.0f ms\n\n",  (((sum_RT[0][1]/RT_count[0][1])+(sum_RT[1][1]/RT_count[1][1]))/2)*10  - (((sum_RT[0][0]/RT_count[0][0])+(sum_RT[1][0]/RT_count[1][0]))/2)*10);
    
	/* Close data file */
	fclose(LRP_OUT);
    fclose(ATTN1_OUT);
    fclose(ATTN2_OUT);
    fclose(ATTN3_OUT);
    
}


/************************************ RUN_TRIAL FUNCTION *************************************/

void run_trial(int target, int condition, float trial_results[3])
{
    float input[NUM_UNITS+2];
    int i;
    float prime_types[3][NUM_UNITS]=
    {
        0,0,0, 0,0,0, 0,0, 0,0,0,	 		// Just noise in all units
        0,0,0, 0,0,0, ext_rH,ext_rS, 0,0,0, 		// Response priming
        0,0,0, 0,0,0, ext_rH,ext_rS, ext_F,ext_C,ext_F 	// Response and attention priming after warning stimulus
    };
    
    input[NUM_UNITS]=target;
    input[NUM_UNITS+1]=condition;
    
    // First clear activations from last trial
    for(i=0; i<NUM_UNITS; i++)
    {
        ext_input[i]=net_input[i]=0;
        activation[i]=REST_ACT;
    }
    
    for(i=0; i<RUN_CYCLES; i++)
        LRP_signal[i]=corrRAct[i]=incorrRAct[i];
    
    // Then run the target priming: this is the interval between warning stimulus and Eriksen stimulus
    for(i=0; i<NUM_UNITS; i++)
        input[i]=prime_types[units_to_prime][i];
    
    run_cycles(input, prime_cycles, 0, trial_results);
    
    // Then present the stimulus
    for(i=0; i<NUM_UNITS; i++)
        input[i]=trial_types[target][condition][i];
    run_cycles(input, RUN_CYCLES, 1, trial_results);
}


/************************************ RUN_CYCLES FUNCTION *************************************/

void run_cycles(float the_inputs[NUM_UNITS+2], int max_cycles, int primeOrReal, float trial_results[3])
{
    int i, j, cycle, response, RT, target, condition;
    int units_to_update[2][NUM_UNITS] =
    {
        0,0,0, 0,0,0, 1,1, 1,1,1,
        //0,0,0, 0,0,0, 1,1, 0,0,0,
        1,1,1, 1,1,1, 1,1, 1,1,1
    };
    
    target=(int) the_inputs[NUM_UNITS];
    condition=(int) the_inputs[NUM_UNITS+1];
    response=NONE;
    
    for(cycle=0; cycle<max_cycles; cycle++)
    {
        // Set the external inputs -- edited by RdK
        for(i=0; i<6; i++)
        {
            if(cycle<onset_delay && (i<LEFT || i>RIGHT))
                ext_input[i]=0;
            else ext_input[i]=the_inputs[i];
        }
        
        
        for(i=6; i<NUM_UNITS; i++)
        {
            if(cycle<attn_delay && (i<LEFT || i>RIGHT))
                ext_input[i]=0;
            else ext_input[i]=the_inputs[i];
        }
        
        
        
        cont_proc=(int) CONT_PROC +(0.5*getNoise());  //DIT MET COND-PROC SNAP IK NOG NIET
        if(response!=NONE && cycle>(RT+cont_proc))
            for(i=0; i<NUM_UNITS; i++)
                ext_input[i]=0;
        
        // Calculate the net input to each unit
        for(i=0; i<NUM_UNITS; i++)
        {
            if(units_to_update[primeOrReal][i]==TRUE)
            {
                new_input[i]=ext_input[i]*estr;
                for(j=0; j<NUM_UNITS; j++)
                {
                    if(activation[j]>0)
                        new_input[i]+=(activation[j]*weights[j][i]);
                }
                
                if(cycle<onset_delay && (i<LEFT || i>RIGHT)) 	//HIER HEB IK WAT CODE DIE 						//UITGESCHAKELD WAS WEER INGESCHAKELD
                    net_input[i]=0;
                else
                    net_input[i]=new_input[i];
                
                if(attentionType==2 && i>7 && i<11) // RdK: low attention input during warning period
                {
                    net_input[i]*=1;
                }

                
                net_input[i]+=noise_val*getNoise();
                net_input[i]*=unit_gain[i];
            }
        }
        
        // Calculate activations using IAC function
        for(i=0; i<NUM_UNITS; i++)
        {
            if(net_input[i]>0)
                activation[i]+=(((MAX_ACT-activation[i])*net_input[i])-((activation[i]-REST_ACT)*DECAY));
            else activation[i]+=(((activation[i]-MIN_ACT)*net_input[i])-((activation[i]-REST_ACT)*DECAY));
            
            if(activation[i]>MAX_ACT)
                activation[i]=MAX_ACT;
            if(activation[i]<MIN_ACT)
                activation[i]=MIN_ACT;
        }
        
        if(attentionType==1) // RdK: fixed values
        {
            activation[8]=-0.06+noise_val*getNoise();
            activation[9]=0.35+noise_val*getNoise();
            activation[10]=-0.06+noise_val*getNoise();
        }
        
        if(primeOrReal==1)
        {
            corrRAct[cycle]=activation[LEFT+target];
            incorrRAct[cycle]=activation[LEFT+(1-target)];
            LRP_signal[cycle]=corrRAct[cycle]-incorrRAct[cycle];
            leftATTNact[cycle]=activation[8];   // these three added by RdK
            ctrATTNact[cycle]=activation[9];
            rightATTNact[cycle]=activation[10];
        }
        
        // See if a response has been generated
		if(response==NONE && primeOrReal==1)
		{
			if((activation[LEFT]>threshold_H && target==H) || (activation[RIGHT]>threshold_S && target==S))
			{
				response=CORRECT;
				RT=cycle;
			}
			else if((activation[LEFT]>threshold_H && target==S) || (activation[RIGHT]>threshold_S && target==H))
			{
				response=ERROR;
				RT=cycle;
			}
		}
        
	} // end of cycle loop
    
	if(primeOrReal==1)
	{
		trial_results[0]=RT;
		trial_results[1]=response;
		trial_results[2]=ext_C;
	}
}



/************************************ GETNOISE FUNCTION *************************************/

float getNoise ()		/* returns gaussian-distributed values, mean 0, sdev 1 -- not yet scaled */
{
	float ran1, ran2, output;
	float fac, rsq, v1, v2;
	do
	{
		ran1=(float) rand()/RAND_MAX;
		ran2=(float) rand()/RAND_MAX;
		v1=(2.0*ran1)-1;
		v2=(2.0*ran2)-1;
		rsq=(v1*v1)+(v2*v2);
	} while (rsq>= 1.0 || rsq == 0.0);
    
	fac=sqrt(-2.0*log(rsq)/rsq);
	output=v1*fac;
	if(fpclassify(output)==FP_NAN)
		printf("Bad random number generated!!.\n");
    
	return output;
}

int irand(int n)
{
	return rand()%n;
}


