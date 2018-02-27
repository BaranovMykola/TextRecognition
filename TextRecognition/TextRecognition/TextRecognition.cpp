#include "TextRecognition.h"
#include "TrainANN.h"

using namespace cv;

std::string recognize(cv::Mat & img)
{
	auto processed = letterHighligh(img);
	processed = rotate(processed, findSkew(processed));
	auto original = processed.clone();
	int s;
	processed = closeCharacters(processed,&s);

	auto chars = sortCharacters(processed.clone());
	auto spaces = segmentWords(processed.clone());

	setlocale(LC_CTYPE, "ukr");
	std::string letters = "¿¡¬√ƒ≈™∆«»≤Ø… ÀÃÕŒœ–—“”‘’÷◊ÿŸ‹ﬁﬂ";
	std::string text;
	int line = chars.size();

	auto mlp = loadANN(ClassifierPrefix + "28x28_MLP_H0");
	
	for (int i = 0; i < line; i++)
	{
		std::vector<cv::Rect> curRow;
		for (auto j : chars)
		{
			if (j.second == i)
			{
				curRow.push_back(j.first);
			}
			std::sort(curRow.begin(), curRow.end(), [](auto l, auto r)
			{
				return l.x < r.x;
			});
		}
		int ch = 0;
		for (int j = 0; j < curRow.size(); j++)
		{
			curRow[j].y -= s;
			Mat let = original(curRow[j]);
			Mat letbin;
			threshold(let, letbin, 127, 255, THRESH_BINARY);
			Mat letbinres;
			cv::resize(letbin, letbinres, Size(28, 28), 0, 0, INTER_LINEAR);
			threshold(letbinres, letbinres, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);

			auto vec = convertMatToVec(letbinres);
			Mat res;
			mlp->predict(vec, res);
			//svm->predict(vec, res);
			float* row = res.ptr<float>(0);
			auto max = std::max_element(row, row + 33);
			int d = std::distance(row, max);
			if (d == 10 && let.rows / let.cols < 2)
			{

			}
			else
			{
				text += letters.substr(d, 1);
			}
			auto sp = *(std::next(spaces.begin(), i));
			for (auto c : sp.second)
			{
				if (j!= curRow.size()-1 && curRow[j].x+curRow[j].width < c && curRow[j + 1].x > c)
				{
					text += "_";
				}
			}
			++ch;
		}
		text += "\n";
	}
	std::cout << letters << std::endl;
	std::cout << text << std::endl;
	return text;
}
