 
void multiply(char *num1, char *num2, char *result, int la, int lb)
{
    char c[NUMWORDS];
    char temp[NUMWORDS];
    int i,j,k=0,x=0,y;
    long int r=0;
    long sum = 0;

    for(i=lb;i>=0;i--){
         r=0;
         for(j=la;j>=0;j--){
             temp[k++] = (num2[i]*num1[j] + r)%10;
             r = (num2[i]*num1[j]+r)/10;
         }
         temp[k++] = r;
         x++;
         for(y = 0;y<x;y++){
             temp[k++] = 0;
         }
    }

    k=0;
    r=0;
    for(i=0;i<la+lb+2;i++){
         sum =0;
         y=0;
         for(j=1;j<=lb+1;j++){
             if(i <= la+j){
                 sum = sum + temp[y+i];
             }
             y += j + la + 1;
         }
         c[k++] = (sum+r) %10;
         r = (sum+r)/10;
    }
    c[k] = r;
    j=0;
    for(i=k-1;i>=0;i--){
         result[j++]=c[i] + 48;
    }
    result[j]='\0';
}

void decrement(char* num1, int length)
{
	for (;length > 0;length--)
	{
		if (num1[length] == 0)
		{
			num1[length] = 9;
			continue;
		}
		num1[length]--;
		return;
	}
}

void rebalance(char* num, int *l)
{
    char reb[NUMWORDS];
    int i = 0;
    for (; i < *l; i++)
    {
        if (num[i] == 0)
        {
            (*l)--;
        }
        else
        {
            break;
        }
    }
    for (int j = i; j < *l; j++)
    {
        reb[j-i] = num[j];
    }
    num = reb;
}

void increment(char* num1, int *length)
{
    for (int i = *length;i > 0;i--)
    {
        if (num1[i] == 9)
        {
            num1[i] = 0;
            continue;
        }
        num1[i]++;
        return;
    }
}

int get_length_discount_prec_null(char *num, int l)
{
    for (int i = 0; i < l; i++)
    {
        if (num[i] == 0)
        {
            l--;
        }
        else
        {
            break;
        }
    }
    return l;
}

char* division(char* num, char* devisor, int n_l, int d_l)
{
    char result[NUMWORDS];
    int res_os = n_l;
    if (d_l > n_l) { return num; }
    char temp[NUMWORDS];
    for (int i = 0; i < NUMWORDS; i++) { temp[i] = num[i]; }
    
    while (get_length_discount_prec_null(temp, n_l) > get_length_discount_prec_null(devisor, d_l))
    {
        //CONTINUE FROM HERE
    }
}

int ModuloByDigits(int previousValue, int modulo)
{
    return ((previousValue * 10) % modulo);
}

char* reverse_string(char *str)
{
    /* skip null */
    if (str == 0)
    {
        exit(1);
    }
    
    /* skip empty string */
    if (*str == 0)
    {
        exit(1);
    }
    
    /* get range */
    char *start = str;
    char *end = start + strlen(str) - 1; /* -1 for \0 */
    char temp;
    
    /* reverse */
    while (end > start)
    {
        /* swap */
        temp = *start;
        *start = *end;
        *end = temp;
        
        /* move */
        ++start;
        --end;
    }
    return str;
}

__kernel void rsa_enc(__global char* p, __global char* q, __global char* message, __global char* result, int l_p, int l_q, int l_msg)
{
	//Compute n
	char n[NUMWORDS];
	multiply(p, q, n, l_p, l_q);

	//Compute totient (ø)
	char t[NUMWORDS];
	decrement(p, &l_p);
	decrement(q, &l_q);
	multiply(p, q, t, l_p, l_q);
}