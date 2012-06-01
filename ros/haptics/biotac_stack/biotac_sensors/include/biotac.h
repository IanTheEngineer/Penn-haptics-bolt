#ifndef BIOTAC_H_
#define BIOTAC_H_

#define MAX_BIOTACS_PER_CHEETAH							3 // should be either 3 or 5

//=========================================================================
// DEFAULT CONSTANTS
//=========================================================================
#define BT_SPI_BITRATE_KHZ_DEFAULT 						4400
#define	BT_AFTERSAMPLE_DELAY_DEFAULT 					50000 		/* Delay after sampling command */
#define BT_INTERWORD_DELAY_DEFAULT						10000		/* Delay between words in communication */
#define BT_SAMPLE_RATE_HZ_DEFAULT						4400
#define BT_FRAMES_IN_BATCH_DEFAULT						5
#define BT_BATCH_MS_DEFAULT								50
#define BT_FRAME_STRUCTURE_DEFAULT  					{\
														BT_E01_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E02_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E03_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E04_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E05_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E06_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E07_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E08_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E09_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E10_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E11_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E12_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E13_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E14_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E15_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E16_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E17_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_E18_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_E19_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_PDC_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, \
														BT_TAC_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, BT_TDC_SAMPLING_COMMAND, BT_PAC_SAMPLING_COMMAND, '\0'}
#ifndef BOOL
typedef int			 									BOOL;
#endif
#define PARITY_GOOD										(BOOL)0
#define PARITY_BAD										(BOOL)!PARITY_GOOD


//==================================================//
// SPI Chip Select control signal //
//==================================================//
#define CS_ALL_BT										0x07					//0b00000111
#define CS_NONE_BT										0x00					//0b00000000
#define CS_BT1											0x01					//0b00000001
#define CS_BT2											0x02					//0b00000010
#define CS_BT3											0x04					//0b00000100
#if MAX_BIOTACS_PER_CHEETAH == 5
	#define CS_BT4										0x05					//0b00000101
	#define CS_BT5										0x06					//0b00000110
#endif

//==================================================//
// SPI version 2 commands define //
//==================================================//
#define COMMAND_LENGTH_BYTE								2
// sampling command //
#define SAMPLE_COMMAND									0x80					//0b10000000
#define BT_PAC_SAMPLING									0						// // command index: 00
#define BT_PDC_SAMPLING									1						// // command index: 01
#define BT_TAC_SAMPLING									2						// // command index: 02
#define BT_TDC_SAMPLING									3						// // command index: 03

#define BT_E01_SAMPLING									17						// // command index: 17
#define BT_E02_SAMPLING									18						// // command index: 18
#define BT_E03_SAMPLING									19						// // command index: 19
#define BT_E04_SAMPLING									20						// // command index: 20
#define BT_E05_SAMPLING									21						// // command index: 21
#define BT_E06_SAMPLING									22						// // command index: 22
#define BT_E07_SAMPLING									23						// // command index: 23
#define BT_E08_SAMPLING									24						// // command index: 24
#define BT_E09_SAMPLING									25						// // command index: 25
#define BT_E10_SAMPLING									26						// // command index: 26
#define BT_E11_SAMPLING									27						// // command index: 27
#define BT_E12_SAMPLING									28						// // command index: 28
#define BT_E13_SAMPLING									29						// // command index: 29
#define BT_E14_SAMPLING									30						// // command index: 30
#define BT_E15_SAMPLING									31						// // command index: 31
#define BT_E16_SAMPLING									32						// // command index: 32
#define BT_E17_SAMPLING									33						// // command index: 33
#define BT_E18_SAMPLING									34						// // command index: 34
#define BT_E19_SAMPLING									35						// // command index: 35

#define BT_PAC_SAMPLING_COMMAND							0x80					//0b10000000 				// command index: 00
#define BT_PDC_SAMPLING_COMMAND							0x83					//0b10000011				// command index: 01
#define BT_TAC_SAMPLING_COMMAND							0x85					//0b10000101				// command index: 02
#define BT_TDC_SAMPLING_COMMAND							0x86					//0b10000110				// command index: 03

#define BT_HAL_SAMPLING_COMMAND							0x9E					//0b10011110				// command index: 15

#define BT_E01_SAMPLING_COMMAND							0xA2					//0b10100010				// command index: 17
#define BT_E02_SAMPLING_COMMAND							0xA4					//0b10100100				// command index: 18
#define BT_E03_SAMPLING_COMMAND							0xA7					//0b10100111				// command index: 19
#define BT_E04_SAMPLING_COMMAND							0xA8					//0b10101000				// command index: 20
#define BT_E05_SAMPLING_COMMAND							0xAB					//0b10101011				// command index: 21
#define BT_E06_SAMPLING_COMMAND							0xAD					//0b10101101				// command index: 22
#define BT_E07_SAMPLING_COMMAND							0xAE					//0b10101110				// command index: 23
#define BT_E08_SAMPLING_COMMAND							0xB0					//0b10110000				// command index: 24
#define BT_E09_SAMPLING_COMMAND							0xB3					//0b10110011				// command index: 25
#define BT_E10_SAMPLING_COMMAND							0xB5					//0b10110101				// command index: 26
#define BT_E11_SAMPLING_COMMAND							0xB6					//0b10110110				// command index: 27
#define BT_E12_SAMPLING_COMMAND							0xB9					//0b10111001				// command index: 28
#define BT_E13_SAMPLING_COMMAND							0xBA					//0b10111010				// command index: 29
#define BT_E14_SAMPLING_COMMAND							0xBC					//0b10111100				// command index: 30
#define BT_E15_SAMPLING_COMMAND							0xBF					//0b10111111				// command index: 31
#define BT_E16_SAMPLING_COMMAND							0xC1					//0b11000001				// command index: 32
#define BT_E17_SAMPLING_COMMAND							0xC2					//0b11000010				// command index: 33
#define BT_E18_SAMPLING_COMMAND							0xC4					//0b11000100				// command index: 34
#define BT_E19_SAMPLING_COMMAND							0xC7					//0b11000111				// command index: 35

// resend command //
#define BT_RESEND_COMMAND								0x20					//0b00100000

// read command //
#define BT_READ_COMMAND									0x61					//0b01100001

	#define BT_FLEX_VERSION_READ_COMMAND				0x10					//0b00010000
	#define BT_FLEX_VERSIOM_READ_LENGTH					2

	#define BT_FIRMWARE_VERSION_READ_COMMAND			0x13					//0b00010011
	#define BT_FIRMWARE_VERSION_READ_LENGTH				4

	#define BT_SERIAL_NUMBER_READ_COMMAND				0x15					//0b00010101
	#define BT_SERIAL_NUMBER_READ_LENGTH				16

	#define BT_CPU_SPEED_READ_COMMAND					0x61					//0b01100001
	#define BT_CPU_SPEED_READ_LENGTH					2

	#define BT_SAMPLING_FREQUENCY_READ_COMMAND			0x70					//0b01110000
	#define BT_SAMPLING_FREQUENCY_READ_LENGTH			2

	#define BT_SAMPLING_PATTERN_READ_COMMAND			0x73					//0b01110011
	#define BT_SAMPLING_PATTERN_READ_LENGTH				44

#define BT_READ_PROPERTY_COMMAND_ARRAY 					{\
                                                        BT_FIRMWARE_VERSION_READ_COMMAND, \
														BT_FLEX_VERSION_READ_COMMAND, \
                                                        BT_FIRMWARE_VERSION_READ_COMMAND, \
														BT_SERIAL_NUMBER_READ_COMMAND, 	\
														BT_CPU_SPEED_READ_COMMAND}

//==================================================//
// SPI version 2 Error Code define //
//==================================================//
#define BT_ERROR_PARITY_CHECK							0xA545					// 0b10100101 01000101, "EE"
#define BT_ERROR_UNKNOWN_COMMAND						0xA53F					// 0b10100101 00111111, "E?"
#define BT_ERROR_INSUFFICIENT_DELAY						0xA52E					// 0b10100101 00101110, "E."
#define BT_ERROR_INSUFFICIENT_SAMPLING_DELAY			0xA52D					// 0b10100101 00101101, "E-"
#define BT_ERROR_CHANNEL_NOT_RECOGNIZED					0xA558					// 0b10100101 01011000, "EX"
#define BT_ERROR_NO_DATA_TO_RESEND						0xA55F					// 0b10100101 01011111, "E_"
#define BT_ERROR_PARAMETER_IS_READ_ONLY					0xA552					// 0b10100101 01010010, "ER"
#define BT_ERROR_ERROR_WRITING_DATA						0xA557					// 0b10100101 01010111, "EW"


#define BT_OK                                0
#define BT_WRONG_NUMBER_ASSIGNED            -1
#define BT_NO_BIOTAC_DETECTED               -2
#define BT_WRONG_MAX_BIOTAC_NUMBER          -3
#define BT_DATA_SIZE_TOO_SMALL              -4
#define BT_NO_CHEETAH_DETECTED              -5
#define BT_UNABLE_TO_OPEN_CHEETAH           -6
#define BT_UNABLE_TO_OPEN_FILE              -7

#define PRINT_ON                             0

//================
// Necessary Includes
//================
#include "cheetah.h"

//==================================================//
// Data structure definition //
//==================================================//
typedef int BioTac;

typedef struct
{
	int index;
	double time;
	double frame_index;
	double batch_index;
	u08 channel_id;
	union
	{
		u16 word;
		u08 byte[2];
	} d[MAX_BIOTACS_PER_CHEETAH];
	u08 bt_parity[MAX_BIOTACS_PER_CHEETAH];
} bt_data;

typedef struct
{
    u08 flex_version[100];
	u08 firmware_version[100];
	u08 serial_number[100];
	union
	{
		u16 value;
		u08 byte[BT_CPU_SPEED_READ_LENGTH];
	} cpu_speed;
	BOOL bt_connected;
} bt_property;

typedef struct
{
	int spi_clock_speed;
	int number_of_biotacs;
	int sample_rate_Hz;
	struct
	{
		int frame_type;
		int frame_size;
		char frame_structure[100];
	} frame;
	struct
	{
		int batch_frame_count;
		int batch_ms;
	} batch;
} bt_info;


//==================================================//
// Print Data Definitions //
//==================================================//
#define YES			(BOOL)1
#define NO			(BOOL)0


//==================================================//
// Function Prototypes //
//==================================================//
BioTac 	bt_cheetah_initialize(const bt_info *biotac, Cheetah *ch_handle);
BioTac 		bt_cheetah_get_properties(Cheetah ch_handle, int bt_select, bt_property *property);
BioTac 		bt_cheetah_configure_batch(Cheetah ch_handle, bt_info *biotac, int num_samples);
bt_data*	bt_configure_save_buffer(int num_samples);
void 		bt_cheetah_collect_batch(Cheetah ch_handle, const bt_info *biotac, bt_data *data, BOOL print_flag);
void		bt_display_errors(BioTac bt_err_code);
BioTac  bt_save_buffer_data(const char *file_name, const bt_data *data, int num_samples);
void 		bt_cheetah_close(Cheetah ch_handle);

#endif /* BIOTAC_H_ */
