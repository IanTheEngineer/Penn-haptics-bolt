/*=========================================================================
| (c) 2011-2012  SynTouch LLC
|--------------------------------------------------------------------------
| Project : BioTac C Library for Cheetah
| File    : biotac.c
| Authors : Gary Lin (gary.lin@syntouchllc.com)
|			Tomonori Yamamoto (tomonori.yamamoto@syntouchllc.com)
|			Jeremy Fishel (jeremy.fishel@syntouchllc.com)
| Modified: Ian McMahon (University of Pennsylvania)
|--------------------------------------------------------------------------
| Function: BioTac-Cheetah communication functions
|--------------------------------------------------------------------------
| Redistribution and use of this file in source and binary forms, with
| or without modification, are permitted.
|
| THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
| "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
| LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
| FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
| COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
| INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
| BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
| LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
| CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
| LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
| ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
| POSSIBILITY OF SUCH DAMAGE.
 ========================================================================*/

#define DEFAULT_TIMER
//#define MACHINE_TIMER

//=========================================================================
// INCLUDES
//=========================================================================
#ifdef _WIN32
#define _CRT_SECURE_NO_DEPRECATE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined (__linux__) || defined (__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <Windows.h>
#endif

#include "biotac_sensors/biotac.h"

//=========================================================================
// CONSTANTS
//=========================================================================
static const unsigned char parity_values[] = \
		{ 0x01, 0x02, 0x04, 0x07, 0x08, 0x0B, 0x0D, 0x0E, \
		  0x10, 0x13, 0x15, 0x16, 0x19, 0x1A, 0x1C, 0x1F, \
		  0x20, 0x23, 0x25, 0x26, 0x29, 0x2A, 0x2C, 0x2F, \
		  0x31, 0x32, 0x34, 0x37, 0x38, 0x3B, 0x3D, 0x3E, \
		  0x40, 0x43, 0x45, 0x46, 0x49, 0x4A, 0x4C, 0x4F, \
		  0x51, 0x52, 0x54, 0x57, 0x58, 0x5B, 0x5D, 0x5E, \
		  0x61, 0x62, 0x64, 0x67, 0x68, 0x6B, 0x6D, 0x6E, \
		  0x70, 0x73, 0x75, 0x76, 0x79, 0x7A, 0x7C, 0x7F, \
		  0x80, 0x83, 0x85, 0x86, 0x89, 0x8A, 0x8C, 0x8F, \
		  0x91, 0x92, 0x94, 0x97, 0x98, 0x9B, 0x9D, 0x9E, \
		  0xA1, 0xA2, 0xA4, 0xA7, 0xA8, 0xAB, 0xAD, 0xAE, \
		  0xB0, 0xB3, 0xB5, 0xB6, 0xB9, 0xBA, 0xBC, 0xBF, \
		  0xC1, 0xC2, 0xC4, 0xC7, 0xC8, 0xCB, 0xCD, 0xCE, \
		  0xD0, 0xD3, 0xD5, 0xD6, 0xD9, 0xDA, 0xDC, 0xDF, \
		  0xE0, 0xE3, 0xE5, 0xE6, 0xE9, 0xEA, 0xEC, 0xEF, \
		  0xF1, 0xF2, 0xF4, 0xF7, 0xF8, 0xFB, 0xFD, 0xFE};

static const char *command_name[] = \
		{ "PAC", "PDC", "TAC", "TDC", "   ", "   ", "   ", "   ", "   ", "   ", \
		  "   ", "   ", "   ", "   ", "   ", "HAL", "REV", "E01", "E02", "E03", \
		  "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13", \
		  "E14", "E15", "E16", "E17", "E18", "E19", "   ", "   ", "   ", "   ", \
		  "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", \
		  "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", "   ", \
		  "   ", "   ", "   ", "   "};

static const char read_command_in_array[] = BT_READ_PROPERTY_COMMAND_ARRAY;
#define NUMBER_OF_READ_COMMAND_IN_ARRAY (int)(sizeof(read_command_in_array)/(sizeof(char)))

#if defined (MACHINE_TIMER)
#if defined (__linux__) || defined (__APPLE__)
static double time_of_frame_start, time_of_frame_end;
static struct timeval tv;
time_t curtime_sec, curtime_usec;
double time_step;
#elif defined (_WIN32)
LARGE_INTEGER frequency;
LARGE_INTEGER time_of_frame_start, time_of_frame_end;
double time_step;
#endif
#endif /* MACHINE_TIMER */

static int count = 0;


//=========================================================================
// INITIAILIZE CHEETAH FOR BIOTAC COMMUNICATION
//=========================================================================
BioTac bt_cheetah_initialize(const bt_info *biotac, Cheetah *ch_handle)
{
	  int mode = 0;
    u16 ports[16];
    u32 unique_ids[16];
    int nelem = 1;

    int i;
    int count;

    // Find all the attached Cheetah devices
  if(PRINT_ON) printf("Searching for Cheetah adapters...\n");
 	count = ch_find_devices_ext(nelem, ports, nelem, unique_ids);
  if(PRINT_ON) printf("%d device(s) found:\n", count);
    // Check the number of Cheetah devices found
    if (count == 0)
    {
    	if(PRINT_ON) printf("Error: No Cheetah devices found!\n");
      return BT_NO_CHEETAH_DETECTED;
    }
    else if (count > nelem)
    {
        // the current version supports only one Cheetah device
    	if(PRINT_ON) printf("WARNING: The current version of software supports one Cheetah device\n");
    	count = nelem;
    }

    for (i = 0; i < count; ++i)
    {
    	// Determine if the device is in-use
        const char *status = "(avail) ";
        if (ports[i] & CH_PORT_NOT_FREE)
        {
        	ports[i] &= ~CH_PORT_NOT_FREE;
            status = "(in-use)";
        }

        // Display device port number, in-use status, and serial number
        if(PRINT_ON) printf("    port=%-3d %s (%04d-%06d)\n",
               ports[i], status,
               unique_ids[i]/1000000,
               unique_ids[i]%1000000);

        // Open the device
        *ch_handle = ch_open(ports[i]);
        if ((*ch_handle) <= 0)
        {
        	if(PRINT_ON) printf("Unable to open Cheetah device on port %d\n", ports[i]);
        	if(PRINT_ON) printf("Error code = %d (%s)\n", (*ch_handle), ch_status_string((*ch_handle)));
        	return BT_UNABLE_TO_OPEN_CHEETAH;
        }

        if(PRINT_ON) printf("Opened Cheetah device on port %d\n", ports[i]);

        if(PRINT_ON) printf("Host interface is %s\n", (ch_host_ifce_speed((*ch_handle))) ? "high speed" : "full speed");

        // Ensure that the SPI subsystem is configured
        ch_spi_configure((*ch_handle), (mode >> 1), mode & 1, CH_SPI_BITORDER_MSB, 0x0);
        if(PRINT_ON) printf("SPI configuration set to mode %d, MSB shift, SS[2:0] active low\n", mode);
        if(PRINT_ON) fflush(stdout);

		ch_target_power((*ch_handle), CH_TARGET_POWER_ON);
		ch_sleep_ms(100);

		// Set the spi_clock_speed
		ch_spi_bitrate((*ch_handle), biotac->spi_clock_speed);
		if(PRINT_ON) printf("Bitrate set to %d kHz\n", biotac->spi_clock_speed);
		if(PRINT_ON) fflush(stdout);
	
		/* PLACEHOLDER FOR FUTURE BIOTAC PROPERTY MEASUREMENTS */

		ch_spi_queue_clear((*ch_handle));
		ch_spi_queue_oe((*ch_handle), 1);
    }

	return BT_OK;
}


//=========================================================================
// GET BIOTAC PROPERTIES
//=========================================================================
BioTac bt_cheetah_get_properties(Cheetah ch_handle, int bt_select, bt_property *property)
{
	int i, len;
	BioTac bt_err_code = BT_OK;
	u08 tmp[100];
	u08 read_command_in_array[] = BT_READ_PROPERTY_COMMAND_ARRAY;
	u08 bt_select_cmd = CS_NONE_BT;

	switch(bt_select)
	{
	case 1:
		bt_select_cmd = CS_BT1;
		break;
	case 2:
		bt_select_cmd = CS_BT2;
		break;
	case 3:
		bt_select_cmd = CS_BT3;
		break;
#if MAX_BIOTACS_PER_CHEETAH == 5
	case 4:
		bt_select_cmd = CS_BT4;
		break;
	case 5:
		bt_select_cmd = CS_BT5;
		break;
#endif
	default:
		bt_err_code = BT_WRONG_NUMBER_ASSIGNED;
		return bt_err_code;
	}

	for(i = 0; i < NUMBER_OF_READ_COMMAND_IN_ARRAY; i++)
	{
		ch_spi_queue_clear(ch_handle);                                          // clean previous commands in Cheetah buffer
		ch_spi_queue_ss(ch_handle, bt_select_cmd);                              // select a BioTac
		ch_spi_queue_byte(ch_handle, 1, BT_READ_COMMAND);                       // queue the read command (1 byte)
		ch_spi_queue_byte(ch_handle, 1, read_command_in_array[i]);              // queue the property command (1 byte)
		ch_spi_queue_ss(ch_handle, CS_NONE_BT);                                 // deselect all the Biotacs
		ch_spi_queue_delay_ns(ch_handle, BT_AFTERSAMPLE_DELAY_DEFAULT);         // delay for BioTac processing the command
        ch_spi_batch_shift(ch_handle, 2, tmp);									// send out a 2-byte command

        len = 0;

		switch(read_command_in_array[i])
		{
		case BT_FLEX_VERSION_READ_COMMAND:
			do {
					ch_spi_queue_clear(ch_handle);
					ch_spi_queue_ss(ch_handle, bt_select_cmd);
					ch_spi_queue_byte(ch_handle, 1, 0);
					ch_spi_batch_shift(ch_handle, 1, &(property->flex_version[len]));
					len += 1;
					if(len > 100)
					{
						break;
					}
			} while (property->flex_version[len-1] != '\0' && property->flex_version[len-1] != 0xFF);
			ch_spi_queue_clear(ch_handle);
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_batch_shift(ch_handle, 0, tmp);
			break;
		case BT_FIRMWARE_VERSION_READ_COMMAND:
			do {
					ch_spi_queue_clear(ch_handle);
					ch_spi_queue_ss(ch_handle, bt_select_cmd);
					ch_spi_queue_byte(ch_handle, 1, 0);
					ch_spi_batch_shift(ch_handle, 1, &(property->firmware_version[len]));
					len += 1;
					if(len > 100)
					{
						break;
					}
			} while (property->firmware_version[len-1] != '\0' && property->firmware_version[len-1] != 0xFF);
			ch_spi_queue_clear(ch_handle);
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_batch_shift(ch_handle, 0, tmp);
			break;
		case BT_SERIAL_NUMBER_READ_COMMAND:
			do {
					ch_spi_queue_clear(ch_handle);
					ch_spi_queue_ss(ch_handle, bt_select_cmd);
					ch_spi_queue_byte(ch_handle, 1, 0);
					ch_spi_batch_shift(ch_handle, 1, &(property->serial_number[len]));
					len += 1;
					if(len > 100)
					{
						break;
					}
			} while (property->serial_number[len-1] != '\0' && property->serial_number[len-1] != 0xFF);
			ch_spi_queue_clear(ch_handle);
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_batch_shift(ch_handle, 0, tmp);
			break;
		case BT_CPU_SPEED_READ_COMMAND:
			ch_spi_queue_clear(ch_handle);
			ch_spi_queue_ss(ch_handle, bt_select_cmd);
			ch_spi_queue_byte(ch_handle, 2, 0);
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_batch_shift(ch_handle, 2, tmp);
			property->cpu_speed.value = tmp[0] * 256 + tmp[1];
			break;
		default:
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			bt_err_code = BT_ERROR_UNKNOWN_COMMAND;
			return bt_err_code;
		}
	}

	// Print out properties of the BioTac(s)
	if(PRINT_ON) printf("\n------- BioTac %d -------\n", bt_select);

	if (property->serial_number[0] == 'B' && property->serial_number[1] == 'T')
	{
		property->bt_connected = YES;
		if(property->firmware_version[0] == '0' && property->firmware_version[1] == '2')
		{
			if(PRINT_ON) printf("Flex Version:\t\t %c.%c\n", property->flex_version[0],  property->flex_version[1]);
            if(PRINT_ON) printf("Firmware Version:\t %c.%c%c\n", \
            		property->firmware_version[1], \
					property->firmware_version[2], property->firmware_version[3]);
			if(PRINT_ON) printf("Serial Number:\t\t %c%c-%c%c-%c%c.%c.%c-%c-%c%c-%c-%c%c%c%c\n", \
					property->serial_number[0], property->serial_number[1], \
                    property->serial_number[2], property->serial_number[3], \
                    property->serial_number[4], property->serial_number[5], \
                    property->serial_number[6], property->serial_number[7], \
                    property->serial_number[8], property->serial_number[9], \
                    property->serial_number[10], property->serial_number[11], \
                    property->serial_number[12], property->serial_number[13], \
                    property->serial_number[14], property->serial_number[15]);
			if(PRINT_ON) printf("CPU Speed:\t\t %.1f MIPS\n", (double)(property->cpu_speed.value/(double)1000));
		}
		else
		{
			if(PRINT_ON) printf("Flex Version:\t\t %s\n", property->flex_version);
            if(PRINT_ON) printf("Firmware Version:\t %s\n", property->firmware_version);
            if(PRINT_ON) printf("Serial Number:\t\t %s\n", property->serial_number);
            if(PRINT_ON) printf("CPU Speed:\t\t %.1f MIPS\n", (double)(property->cpu_speed.value/(double)1000));
		}
	}
	else
	{
		property->bt_connected = NO;

		if(PRINT_ON) printf("Flex Version:\t\t %s\n", "N/A");
		if(PRINT_ON) printf("Software Version:\t %s\n", "N/A");
		if(PRINT_ON) printf("Serial Number:\t\t %s\n", "N/A");
		if(PRINT_ON) printf("CPU Speed:\t\t %s\n", "N/A");
	}

	return bt_err_code;
}

//=========================================================================
// CONFIGURATION OF BATCH
//=========================================================================
BioTac bt_cheetah_configure_batch(Cheetah ch_handle, bt_info *biotac, int num_samples)
{
	BioTac bt_err_code = BT_OK;
	int i, j, k;
	int bt_select_cmd;
	int words_per_sample = MAX_BIOTACS_PER_CHEETAH + 1;
	int cs_per_sample = MAX_BIOTACS_PER_CHEETAH + 3;
	double additional_delay = \
			(1/(double)biotac->sample_rate_Hz)*1000000000 \
			- ((words_per_sample*16 + cs_per_sample*8) / (double)biotac->spi_clock_speed)*1000000 \
			- BT_AFTERSAMPLE_DELAY_DEFAULT - BT_INTERWORD_DELAY_DEFAULT - 3000;  	// in nano second
	const char frame_structure_tmp[] = BT_FRAME_STRUCTURE_DEFAULT;

	// Set frame structure and frame count (a future version may support change of frame structures while running the program) 
	strcpy(biotac->frame.frame_structure, frame_structure_tmp);
	biotac->frame.frame_size = (int)(sizeof(frame_structure_tmp)/(sizeof(char)) - 1); // -1: subtract the null character
	biotac->batch.batch_frame_count = (int)(biotac->batch.batch_ms / \
		(double)((1/(double)biotac->sample_rate_Hz) * (biotac->frame.frame_size) * 1000));

	// Check if num_samples is long enough
	if(num_samples < ((biotac->frame.frame_size) * (biotac->batch.batch_frame_count)))
	{
		bt_err_code = BT_DATA_SIZE_TOO_SMALL;
		return bt_err_code;
	}

	// Configure BioTac sampling
	ch_spi_queue_clear(ch_handle);

	for (i = 0; i < biotac->batch.batch_frame_count; i++)
	{
		for (j = 0; j < biotac->frame.frame_size; j++)
		{
			ch_spi_queue_ss(ch_handle, CS_ALL_BT);
			ch_spi_queue_byte(ch_handle, 1, biotac->frame.frame_structure[j]);
			ch_spi_queue_byte(ch_handle, 1, 0x00);
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_queue_delay_ns(ch_handle, BT_AFTERSAMPLE_DELAY_DEFAULT);
			for (k = 1; k <= MAX_BIOTACS_PER_CHEETAH; k++)
			{
				switch(k)
				{
				case 1:
					bt_select_cmd = CS_BT1;
					break;
				case 2:
					bt_select_cmd = CS_BT2;
					break;
				case 3:
					bt_select_cmd = CS_BT3;
					break;
#if MAX_BIOTACS_PER_CHEETAH == 5
				case 4:
					bt_select_cmd = CS_BT4;
					break;
				case 5:
					bt_select_cmd = CS_BT5;
					break;
#endif
				}
				ch_spi_queue_ss(ch_handle, (u08)bt_select_cmd);
				ch_spi_queue_byte(ch_handle, 2, 0x00);
			}
			ch_spi_queue_ss(ch_handle, CS_NONE_BT);
			ch_spi_queue_delay_ns(ch_handle, BT_INTERWORD_DELAY_DEFAULT);
			ch_spi_queue_delay_ns(ch_handle, (int)additional_delay);
		}
	}

	/**** Time stamps from a computer (by default, it's disabled) ****/
#ifdef MACHINE_TIMER
#if defined (__linux__) || defined (__APPLE__)
	// Get a start time stamp
	gettimeofday(&tv, NULL);
	curtime_sec = tv.tv_sec;
	curtime_usec = tv.tv_usec;

	time_of_frame_start = curtime_sec + (double)curtime_usec/1000000;
#elif defined (_WIN32)
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&time_of_frame_start);
#endif
#endif /* MACHINE_TIMER */

	// Send initial batches
	for (i = 0; i < 16; i++)
	{
		ch_spi_async_submit(ch_handle);
	}

	return bt_err_code;
}


//=========================================================================
// CONFIGURATION OF SAVE BUFFER
//=========================================================================
bt_data* bt_configure_save_buffer(int number_of_samples)
{
	bt_data *data;
	count = 0;
	data = malloc(number_of_samples * sizeof *data);
	return data;
}


//=========================================================================
// COLLECT BATCH
//=========================================================================
void bt_cheetah_collect_batch(Cheetah ch_handle, const bt_info *biotac, bt_data *data, BOOL print_flag)
{
	int i, j;
 	int byte_shift = 2 + (MAX_BIOTACS_PER_CHEETAH * 2);			// 2 bytes of command + 2 bytes per BioTac data
	int spi_data_len;
	int number_of_samples_in_batch;
	u08 *bt_raw_data;

	spi_data_len = ch_spi_batch_length(ch_handle);
	bt_raw_data = malloc(spi_data_len * sizeof *bt_raw_data);
	ch_spi_async_collect(ch_handle, spi_data_len, bt_raw_data);
	ch_spi_async_submit(ch_handle);

	number_of_samples_in_batch = spi_data_len/byte_shift;

	/**** Time stamps from a computer (by default, it's disabled) ****/
#ifdef MACHINE_TIMER
#if defined (__linux__) || defined (__APPLE__)
	gettimeofday(&tv, NULL);
	curtime_sec = tv.tv_sec;
	curtime_usec = tv.tv_usec;

	time_of_frame_end = curtime_sec + (double)curtime_usec/1000000;
	time_step = (time_of_frame_end - time_of_frame_start) / number_of_samples_in_batch;
	time_of_frame_start = time_of_frame_end;
#elif defined (_WIN32)
	QueryPerformanceCounter(&time_of_frame_end);
	time_step = ((time_of_frame_end.QuadPart - time_of_frame_start.QuadPart) / (double)frequency.QuadPart) / number_of_samples_in_batch;
	time_of_frame_start = time_of_frame_end;
#endif
#endif /* MACHINE_TIMER */

	for(i = 0; i < number_of_samples_in_batch; i++)
	{
		if(count != 0)
		{
			if(i==0)
			{
				data[count].batch_index = data[count-1].batch_index + 1;
			}
			else
			{
				data[count].batch_index = data[count-1].batch_index;
			}
			if((i%(biotac->frame.frame_size)) == 0)
			{
				data[count].frame_index = data[count-1].frame_index + 1;
			}
			else
			{
				data[count].frame_index = data[count-1].frame_index;
			}
			/**** Time stamps from a computer (by default, it's disabled) ****/
#ifdef DEFAULT_TIMER
			data[count].time = 0.0;
#elif defined MACHINE_TIMER
			data[count].time = data[count-1].time + time_step;
#endif
		}
		else
		{
			data[count].batch_index = 1;
			data[count].frame_index = 1;
			data[count].time = 0;
		}

		data[count].channel_id = (biotac->frame.frame_structure[i%(biotac->frame.frame_size)] & 0x7E) >> 1;

		for(j = 0; j < MAX_BIOTACS_PER_CHEETAH; j++)
		{
			data[count].d[j].word = (bt_raw_data[i*byte_shift + j*2 + 2] >> 1) * 32 + (bt_raw_data[i*byte_shift + j*2 + 3] >> 3);

			if((parity_values[bt_raw_data[i*byte_shift + j*2 + 2] >> 1] == bt_raw_data[i*byte_shift + j*2 + 2]) && \
					(parity_values[bt_raw_data[i*byte_shift + j*2 + 3] >> 1] == bt_raw_data[i*byte_shift + j*2 + 3]))
			{
				data[count].bt_parity[j] = PARITY_GOOD;
			}
			else
			{
				data[count].bt_parity[j] = PARITY_BAD;
			}
		}

		// Print data on Terminal
		if(print_flag)
		{
			if(PRINT_ON) printf("%8d,  ", count);
			/**** Time stamps from a computer (by default, it only displays NULL) ****/
#ifdef DEFAULT_TIMER
			if(PRINT_ON) printf("%s, ", "NULL");
#elif defined MACHINE_TIMER
			if(PRINT_ON) printf("%3.6f, ", data[count].time);
#endif
			if(PRINT_ON) printf("%6.0f, %6.0f,  %s ", data[count].batch_index, data[count].frame_index, command_name[data[count].channel_id]);
			for(j = 0; j < MAX_BIOTACS_PER_CHEETAH; j++)
			{
				if(PRINT_ON) printf("%6d, %d; ", data[count].d[j].word, data[count].bt_parity[j]);
			}
			if(PRINT_ON) printf("\n");
		}
		count++;
	}
	free(bt_raw_data);
}


//=========================================================================
// SAVE DATA IN A FILE
//=========================================================================
BioTac bt_save_buffer_data(const char *file_name, const bt_data *data, int num_samples)
{
	int i, j;
	FILE *fp;

	fp = fopen(file_name, "w");
	if (!fp)
	{
		if(PRINT_ON) printf("Error: Cannot open output file.\n");
		return BT_UNABLE_TO_OPEN_FILE;
	}

	// Write data in the file. By default, a saved data format is as follows:
	// (time, batch_index, frame_index, channel_id, value[0], bt_parity[0],  value[1], bt_parity[1],  value[2], bt_parity[2], ...)
	for (i = 0; i < num_samples; i++)
	{
		fprintf(fp, "%.6f %.0f %.0f %u ", data[i].time, data[i].batch_index, data[i].frame_index, data[i].channel_id);
//		fprintf(fp, "%.6f %.0f %.0f %s ", data[i].time, data[i].batch_index, data[i].frame_index, command_name[data[i].channel_id]);
		for (j = 0; j < MAX_BIOTACS_PER_CHEETAH; j++)
		{
			fprintf(fp, "%d %d ", data[i].d[j].word, data[i].bt_parity[j]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	if(PRINT_ON) printf("Saved data in %s\n", file_name);
  return BT_OK;
}


//=========================================================================
// PRINT ERROR CODES
//=========================================================================
void bt_display_errors(BioTac bt_err_code)
{
	char *error_str = malloc(100 * sizeof(*error_str));

	switch(bt_err_code)
	{
	case BT_WRONG_NUMBER_ASSIGNED:
		strcpy(error_str, "Wrong BioTac number assigned!");
		break;
	case BT_NO_BIOTAC_DETECTED:
		strcpy(error_str, "No BioTac detected!");
		break;
	case BT_WRONG_MAX_BIOTAC_NUMBER:
		strcpy(error_str, "Wrong maximum number of BioTacs assigned (should be 3 or 5)!");
		break;
	case BT_DATA_SIZE_TOO_SMALL:
		strcpy(error_str, "The number of samples is too small!");
		break;
  case BT_NO_CHEETAH_DETECTED:
    strcpy(error_str, "No Cheetah device detected!");
    break;
  case BT_UNABLE_TO_OPEN_CHEETAH:
    strcpy(error_str, "Unable to open Cheetah device on current port.");
    break;
  case BT_UNABLE_TO_OPEN_FILE:
    strcpy(error_str, "Cannot open output file.");
    break;
  default:
    strcpy(error_str, "Unrecognized Biotac error encountered.");
    break;
	}

	if(PRINT_ON) printf("\nError: %s\n\n", error_str);

	free(error_str);
}


//=========================================================================
// CLOSE CHEETAH CONFIGURATION
//=========================================================================
void bt_cheetah_close(Cheetah ch_handle)
{
	ch_spi_queue_oe(ch_handle, 0);
	ch_close(ch_handle);
}
